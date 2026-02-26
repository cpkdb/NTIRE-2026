import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from datasets import AIGIDataset
from models import TimmClassifier, RINEClassifier, DINOv2Classifier, DINOv3Classifier, DINOv3FreqMoE, FreqClassifier
from transforms import (
    get_train_transform, get_val_transform, get_robust_val_transform,
    get_consistency_transform, random_corrupt_pil,
)
from utils.metrics import compute_roc_auc


def contrastive_loss(features, labels, margin=1.0):
    features = F.normalize(features, dim=1)
    sim = torch.mm(features, features.t())
    labels = labels.float().unsqueeze(1)
    mask_pos = (labels == labels.t()).float()
    mask_neg = 1.0 - mask_pos
    eye = torch.eye(len(labels), device=features.device)
    loss_pos = ((1 - sim) * mask_pos * (1 - eye)).sum() / (mask_pos.sum() - len(labels) + 1e-8)
    loss_neg = (F.relu(sim - margin + 1) * mask_neg).sum() / (mask_neg.sum() + 1e-8)
    return loss_pos + loss_neg


def load_balance_loss(gates):
    avg_gate = gates.mean(dim=0)
    return gates.shape[1] * (avg_gate ** 2).sum()


def build_model(args, device):
    if args.model_type == "rine":
        return RINEClassifier(
            backbone=args.backbone, device=device,
            num_hooks=getattr(args, 'num_hooks', None),
        ).to(device)
    elif args.model_type == "dinov2":
        return DINOv2Classifier(
            num_hooks=getattr(args, 'num_hooks', None),
            lora_layers=getattr(args, 'lora_layers', 0),
        ).to(device)
    elif args.model_type == "dinov3":
        return DINOv3Classifier(
            num_hooks=getattr(args, 'num_hooks', None),
            lora_layers=getattr(args, 'lora_layers', 0),
        ).to(device)
    elif args.model_type == "dinov3_moe":
        model = DINOv3FreqMoE(
            num_experts=getattr(args, 'num_experts', 3),
            num_hooks=getattr(args, 'num_hooks', None),
            lora_layers=getattr(args, 'lora_layers', 0),
        ).to(device)
        return model
    elif args.model_type == "freq":
        return FreqClassifier(backbone_type="imagenet").to(device)
    return TimmClassifier(args.model, pretrained=True).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_contrastive=False, consistency_weight=0.0,
                    label_smoothing=0.0, consistency_loader=None,
                    lb_weight=0.0, epoch=0, total_epochs=1,
                    cons_ramp_epochs=0):
    model.train()
    total_loss = 0
    if getattr(model, "_experts_frozen", False) and hasattr(model, "experts"):
        model.experts.eval()
    cons_iter = iter(consistency_loader) if consistency_loader else None
    has_moe = hasattr(model, 'forward_with_aux')

    effective_cons_weight = consistency_weight * min(1.0, epoch / cons_ramp_epochs) if cons_ramp_epochs > 0 else consistency_weight
    effective_lb = lb_weight * max(0.0, 1.0 - (epoch + 1) / total_epochs) if total_epochs > 0 else lb_weight

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Label smoothing
        smooth_labels = labels.float()
        if label_smoothing > 0:
            smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / 2

        if has_moe and (use_contrastive or effective_lb > 0):
            logits, feats, gates, _ = model.forward_with_aux(images)
            loss_cls = criterion(logits, smooth_labels)
            loss = loss_cls
            if use_contrastive:
                loss = loss + contrastive_loss(feats, labels)
            if effective_lb > 0:
                loss = loss + effective_lb * load_balance_loss(gates)
        elif use_contrastive:
            logits, feats = model.get_features(images)
            loss_cls = criterion(logits, smooth_labels)
            loss_cont = contrastive_loss(feats, labels)
            loss = loss_cls + 1.0 * loss_cont
        else:
            outputs = model(images)
            loss = criterion(outputs, smooth_labels)

        # Consistency loss
        if cons_iter is not None and effective_cons_weight > 0:
            try:
                (clean_imgs, corrupt_imgs), cons_labels = next(cons_iter)
            except StopIteration:
                cons_iter = iter(consistency_loader)
                (clean_imgs, corrupt_imgs), cons_labels = next(cons_iter)

            clean_imgs = clean_imgs.to(device)
            corrupt_imgs = corrupt_imgs.to(device)

            with torch.no_grad():
                clean_logits = model(clean_imgs)
            corrupt_logits = model(corrupt_imgs)

            clean_probs = torch.sigmoid(clean_logits).detach()
            corrupt_probs = torch.sigmoid(corrupt_logits)
            loss_cons = F.binary_cross_entropy(corrupt_probs, clean_probs)
            loss = loss + effective_cons_weight * loss_cons

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in tqdm(loader, desc="Validating"):
        images = images.to(device)
        probs = model.predict_proba(images).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return compute_roc_auc(all_labels, all_probs)


class ConsistencyDataset(torch.utils.data.Dataset):
    """Wraps AIGIDataset to return (clean, corrupted) paired views."""
    def __init__(self, base_dataset, consistency_transform):
        self.base = base_dataset
        self.transform = consistency_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        original_transform = self.base.dataset.transform if hasattr(self.base, 'dataset') else self.base.transform
        # Access underlying image directly
        if hasattr(self.base, 'dataset'):
            img_path, label = self.base.dataset.samples[self.base.indices[idx]]
        else:
            img_path, label = self.base.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        clean, corrupted = self.transform(img)
        return (clean, corrupted), torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--model_type", type=str, default="rine", choices=["timm", "rine", "dinov2", "dinov3", "dinov3_moe", "freq"])
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default="/workspace/experiments")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_robust_aug", action="store_true")
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_hooks", type=int, default=None)
    parser.add_argument("--shards", type=str, default=None, help="Comma-separated shard indices, e.g. '0,1,5'")
    parser.add_argument("--val_shards", type=str, default=None, help="Hold-out shard indices for validation, e.g. '2'")
    parser.add_argument("--lora_layers", type=int, default=0)
    parser.add_argument("--lb_weight", type=float, default=0.0)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--warm_start", type=str, default=None, help="Pretrained DINOv3 checkpoint for warm-start MoE")
    parser.add_argument("--freq_ckpt", type=str, default=None, help="Pretrained FreqClassifier checkpoint")
    parser.add_argument("--freeze_lora_only", action="store_true", help="Freeze LoRA params (Stage A)")
    parser.add_argument("--freeze_experts", action="store_true", help="Freeze expert params (Stage A)")
    parser.add_argument("--freeze_proj", action="store_true", help="Freeze proj1/proj2/alpha params")
    parser.add_argument("--moe_lr_scale", type=float, default=1.0, help="LR scale for router/cls_router_proj/freq_enc")
    parser.add_argument("--lora_lr_scale", type=float, default=1.0, help="LR scale for LoRA params (Stage B)")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--cons_ramp_epochs", type=int, default=3)
    parser.add_argument("--use_gumbel", action="store_true", help="Enable Gumbel-Softmax routing")
    parser.add_argument("--gumbel_start_epoch", type=int, default=2, help="Epoch to start Gumbel-Softmax")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Redirect stdout/stderr to log file in output_dir
    import sys
    log_file = open(output_dir / "train.log", "a")
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data)
        def flush(self):
            for s in self.streams: s.flush()
        def isatty(self): return False
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    backbone_type = "clip" if args.model_type == "rine" else "imagenet"
    train_transform = get_train_transform(robust=not args.no_robust_aug, backbone_type=backbone_type)
    val_transform = get_val_transform(backbone_type=backbone_type)
    robust_val_transform = get_robust_val_transform(backbone_type=backbone_type)
    consistency_transform = get_consistency_transform(backbone_type=backbone_type)

    shards = [int(s) for s in args.shards.split(",")] if args.shards else None
    val_shards = [int(s) for s in args.val_shards.split(",")] if args.val_shards else None

    if val_shards:
        # Hold-out validation: train on --shards, validate on --val_shards
        train_dataset = AIGIDataset(args.data_root, shards=shards, transform=train_transform)
        train_sub = train_dataset
        val_dataset_clean = AIGIDataset(args.data_root, shards=val_shards, transform=val_transform)
        val_dataset_robust = AIGIDataset(args.data_root, shards=val_shards, transform=robust_val_transform)
        val_sub_clean = val_dataset_clean
        val_sub_robust = val_dataset_robust
        print(f"Hold-out validation: train={len(train_dataset)} images, val={len(val_dataset_clean)} images")
    else:
        # Legacy: random split from training shards
        train_dataset = AIGIDataset(args.data_root, shards=shards, transform=train_transform)
        val_size = int(len(train_dataset) * args.val_ratio)
        train_size = len(train_dataset) - val_size
        train_sub, val_sub = random_split(train_dataset, [train_size, val_size])
        val_dataset = AIGIDataset(args.data_root, shards=shards, transform=val_transform)
        val_indices = val_sub.indices
        val_sub_clean = torch.utils.data.Subset(val_dataset, val_indices)
        robust_val_dataset = AIGIDataset(args.data_root, shards=shards, transform=robust_val_transform)
        val_sub_robust = torch.utils.data.Subset(robust_val_dataset, val_indices)

    # Consistency dataset
    cons_dataset = ConsistencyDataset(train_sub, consistency_transform)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_sub_clean, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    robust_val_loader = DataLoader(val_sub_robust, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    cons_loader = DataLoader(cons_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = build_model(args, device)
    use_contrastive = args.model_type in ("rine", "dinov2", "dinov3", "dinov3_moe")

    if args.model_type == "dinov3_moe":
        if args.warm_start:
            print(f"Warm starting MoE from {args.warm_start}")
            model.load_trainable(args.warm_start, device=device)
        if args.freq_ckpt:
            print(f"Loading freq encoder from {args.freq_ckpt}")
            model.load_freq_encoder(args.freq_ckpt, device=device)

    if use_contrastive:
        criterion = nn.BCEWithLogitsLoss()
        if args.model_type == "dinov3_moe" and (args.freeze_lora_only or args.freeze_experts or args.freeze_proj or args.lora_lr_scale != 1.0 or args.moe_lr_scale != 1.0):
            moe_params, lora_params, moe_new_params = [], [], []
            lora_ids = {id(p) for p in model._lora_params}
            expert_ids = {id(p) for p in model.experts.parameters()}
            proj_ids = {id(p) for p in model.proj1.parameters()}
            proj_ids.update(id(p) for p in model.proj2.parameters())
            proj_ids.add(id(model.alpha))
            moe_new_ids = set()
            for p in list(model.router.parameters()) + list(model.cls_router_proj.parameters()) + list(model.freq_enc.parameters()) + [model.tau]:
                moe_new_ids.add(id(p))
            # Freeze pass (independent checks)
            for name, p in model.named_parameters():
                if args.freeze_proj and id(p) in proj_ids:
                    p.requires_grad = False
                if not p.requires_grad:
                    continue
                if args.freeze_lora_only and id(p) in lora_ids:
                    p.requires_grad = False
                if args.freeze_experts and id(p) in expert_ids:
                    p.requires_grad = False
            # Group pass (only trainable params)
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if id(p) in lora_ids:
                    lora_params.append(p)
                elif id(p) in moe_new_ids:
                    moe_new_params.append(p)
                else:
                    moe_params.append(p)
            if args.freeze_experts and hasattr(model, "experts"):
                model._experts_frozen = True

            grouped_ids = {id(p) for g in (moe_params, moe_new_params, lora_params) for p in g}
            expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
            if grouped_ids != expected_ids:
                raise RuntimeError(f"Param grouping mismatch: missing={len(expected_ids - grouped_ids)} leaked={len(grouped_ids - expected_ids)}")

            param_groups = []
            if moe_params:
                param_groups.append({"params": moe_params, "lr": args.lr})
            if moe_new_params:
                param_groups.append({"params": moe_new_params, "lr": args.lr * args.moe_lr_scale})
            if lora_params:
                param_groups.append({"params": lora_params, "lr": args.lr * args.lora_lr_scale})
            if not param_groups:
                raise RuntimeError("No trainable parameters left after freeze settings.")
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=1e-4)
    elif args.model_type == "freq":
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=1e-4)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup_epochs > 0:
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs)),
        ], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    def _trainable_state_dict():
        """Save non-backbone state (keeps LoRA, proj, head/experts even if frozen in Stage A)."""
        return {k: v for k, v in model.state_dict().items()
                if not k.startswith("backbone.") or ".lora_" in k}

    start_epoch = 0
    best_robust_auc = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("trainable_model", ckpt.get("model", {}))
        model.load_state_dict(state, strict=False)
        best_robust_auc = ckpt.get("robust_auc", 0) or 0
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            print(f"Resumed from epoch {start_epoch}, best_robust_auc={best_robust_auc:.4f}")
        except (ValueError, KeyError):
            # Stage switch: keep weights + best_robust_auc, reset epoch counter
            print(f"Optimizer mismatch (stage switch), loaded weights only from epoch {ckpt['epoch']}, best_robust_auc={best_robust_auc:.4f}")

    # Determine stage tag for checkpoint naming
    stage_tag = "b" if (args.resume and not getattr(args, 'freeze_lora_only', False)) else "a"

    writer = SummaryWriter(output_dir / "logs")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        if args.model_type == "dinov3_moe" and hasattr(model, "_use_gumbel"):
            stage_epoch = epoch - start_epoch
            if args.use_gumbel and stage_epoch >= args.gumbel_start_epoch:
                model._use_gumbel = True
                gumbel_span = max(1, args.epochs - start_epoch - args.gumbel_start_epoch - 1)
                gumbel_progress = min(1.0, (stage_epoch - args.gumbel_start_epoch) / gumbel_span)
                model._gumbel_tau = max(0.1, 1.0 - 0.9 * gumbel_progress)
                print(f"Gumbel routing: enabled (tau={model._gumbel_tau:.4f})")
            else:
                model._use_gumbel = False
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_contrastive, consistency_weight=args.consistency_weight,
            label_smoothing=args.label_smoothing, consistency_loader=cons_loader,
            lb_weight=args.lb_weight, epoch=epoch, total_epochs=args.epochs,
            cons_ramp_epochs=args.cons_ramp_epochs,
        )
        scheduler.step()
        val_auc = validate(model, val_loader, device)
        robust_auc = validate(model, robust_val_loader, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("AUC/robust_val", robust_auc, epoch)
        print(f"Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Robust Val AUC: {robust_auc:.4f}")

        # Expert diagnostics for MoE
        if hasattr(model, 'forward_with_aux'):
            model.eval()
            with torch.no_grad():
                sample_imgs = next(iter(val_loader))[0][:16].to(device)
                _, _, gates, _ = model.forward_with_aux(sample_imgs)
                avg_g = gates.mean(dim=0)
                for ei, gv in enumerate(avg_g):
                    writer.add_scalar(f"MoE/expert_{ei}_gate", gv.item(), epoch)
                writer.add_scalar("MoE/gate_entropy", -(avg_g * (avg_g + 1e-8).log()).sum().item(), epoch)
                writer.add_scalar("MoE/tau", model.tau.item(), epoch)

        ckpt = {
            "trainable_model": _trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "robust_auc": robust_auc,
            "val_auc": val_auc,
            "best_robust_auc": best_robust_auc,
        }
        torch.save(ckpt, output_dir / "last.pt")
        torch.save(ckpt, output_dir / f"epoch_{epoch}.pt")

        if robust_auc > best_robust_auc:
            best_robust_auc = robust_auc
            ckpt["best_robust_auc"] = best_robust_auc
            torch.save(ckpt, output_dir / f"best_stage_{stage_tag}.pt")
            print(f"New best Robust AUC: {best_robust_auc:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
