import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets import AIGIDataset
from models import TimmClassifier, RINEClassifier, DINOv2Classifier, DINOv3Classifier, FreqClassifier
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
    elif args.model_type == "freq":
        return FreqClassifier(backbone_type="imagenet").to(device)
    return TimmClassifier(args.model, pretrained=True).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_contrastive=False, consistency_weight=0.0,
                    label_smoothing=0.0, consistency_loader=None):
    model.train()
    total_loss = 0
    cons_iter = iter(consistency_loader) if consistency_loader else None

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Label smoothing
        smooth_labels = labels.float()
        if label_smoothing > 0:
            smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / 2

        if use_contrastive:
            logits, feats = model.get_features(images)
            loss_cls = criterion(logits, smooth_labels)
            loss_cont = contrastive_loss(feats, labels)
            loss = loss_cls + 1.0 * loss_cont
        else:
            outputs = model(images)
            loss = criterion(outputs, smooth_labels)

        # Consistency loss
        if cons_iter is not None and consistency_weight > 0:
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
            loss = loss + consistency_weight * loss_cons

        loss.backward()
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
    parser.add_argument("--model_type", type=str, default="rine", choices=["timm", "rine", "dinov2", "dinov3", "freq"])
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="/workspace/experiments")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_robust_aug", action="store_true")
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_hooks", type=int, default=None)
    parser.add_argument("--shards", type=str, default=None, help="Comma-separated shard indices, e.g. '0,1,5'")
    parser.add_argument("--lora_layers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone_type = "clip" if args.model_type == "rine" else "imagenet"
    train_transform = get_train_transform(robust=not args.no_robust_aug, backbone_type=backbone_type)
    val_transform = get_val_transform(backbone_type=backbone_type)
    robust_val_transform = get_robust_val_transform(backbone_type=backbone_type)
    consistency_transform = get_consistency_transform(backbone_type=backbone_type)

    shards = [int(s) for s in args.shards.split(",")] if args.shards else None
    train_dataset = AIGIDataset(args.data_root, shards=shards, transform=train_transform)
    val_size = int(len(train_dataset) * args.val_ratio)
    train_size = len(train_dataset) - val_size
    train_sub, val_sub = random_split(train_dataset, [train_size, val_size])

    # Clean val
    val_dataset = AIGIDataset(args.data_root, shards=shards, transform=val_transform)
    val_indices = val_sub.indices
    val_sub_clean = torch.utils.data.Subset(val_dataset, val_indices)

    # Robust val
    robust_val_dataset = AIGIDataset(args.data_root, shards=shards, transform=robust_val_transform)
    val_sub_robust = torch.utils.data.Subset(robust_val_dataset, val_indices)

    # Consistency dataset
    cons_dataset = ConsistencyDataset(train_sub, consistency_transform)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_sub_clean, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    robust_val_loader = DataLoader(val_sub_robust, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    cons_loader = DataLoader(cons_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = build_model(args, device)
    use_contrastive = args.model_type in ("rine", "dinov2", "dinov3")

    if use_contrastive:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=1e-4)
    elif args.model_type == "freq":
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=1e-4)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Collect frozen parameter names for efficient filtering
    frozen_keys = {n for n, p in model.named_parameters() if not p.requires_grad}

    def _trainable_state_dict():
        """Only save trainable params (~3MB) instead of full state_dict (~1.2GB).
        Backbone weights are frozen pretrained weights, identical on every init."""
        return {k: v for k, v in model.state_dict().items() if k not in frozen_keys}

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # Support both new (trainable_model) and old (model=full state_dict) formats
        state = ckpt.get("trainable_model", ckpt.get("model", {}))
        model.load_state_dict(state, strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(output_dir / "logs")
    best_robust_auc = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_contrastive, consistency_weight=args.consistency_weight,
            label_smoothing=args.label_smoothing, consistency_loader=cons_loader,
        )
        scheduler.step()
        val_auc = validate(model, val_loader, device)
        robust_auc = validate(model, robust_val_loader, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("AUC/robust_val", robust_auc, epoch)
        print(f"Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Robust Val AUC: {robust_auc:.4f}")

        # Save only trainable params (~3MB) + optimizer + metadata
        ckpt = {
            "trainable_model": _trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, output_dir / "last.pt")

        # Select by robust AUC
        if robust_auc > best_robust_auc:
            best_robust_auc = robust_auc
            torch.save(ckpt, output_dir / "best.pt")
            print(f"New best Robust AUC: {best_robust_auc:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
