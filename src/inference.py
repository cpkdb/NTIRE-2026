import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

from datasets import AIGIDataset, ToyDataset
from models import TimmClassifier, RINEClassifier, DINOv2Classifier, DINOv3Classifier, DINOv3FreqMoE, FreqClassifier
from transforms import get_val_transform, get_tta_transforms, _get_norm


def load_model(model_type, model_name, backbone, checkpoint, device, num_hooks=None, lora_layers=0, num_experts=3):
    if model_type == "rine":
        model = RINEClassifier(backbone=backbone, device=device, num_hooks=num_hooks).to(device)
    elif model_type == "dinov2":
        model = DINOv2Classifier(num_hooks=num_hooks, lora_layers=lora_layers).to(device)
    elif model_type == "dinov3":
        model = DINOv3Classifier(num_hooks=num_hooks, lora_layers=lora_layers).to(device)
    elif model_type == "dinov3_moe":
        model = DINOv3FreqMoE(num_experts=num_experts, num_hooks=num_hooks, lora_layers=lora_layers).to(device)
    elif model_type == "freq":
        model = FreqClassifier(backbone_type="imagenet").to(device)
    else:
        model = TimmClassifier(model_name, pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("trainable_model", ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def inference(model, loader, device):
    results = []
    for images, names in tqdm(loader, desc="Inference"):
        images = images.to(device)
        probs = model.predict_proba(images).cpu().numpy()
        for name, prob in zip(names, probs):
            results.append({"image_name": name, "score": float(prob)})
    return results


@torch.no_grad()
def inference_tta(model, dataset, tta_transforms, batch_size, device):
    results = {}
    for t_idx, tfm in enumerate(tta_transforms):
        dataset.transform = tfm
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        for images, names in tqdm(loader, desc=f"TTA {t_idx+1}/{len(tta_transforms)}"):
            images = images.to(device)
            probs = model.predict_proba(images).cpu().numpy()
            for name, prob in zip(names, probs):
                results.setdefault(name, []).append(float(prob))
    return [{"image_name": k, "score": sum(v) / len(v)} for k, v in results.items()]


@torch.no_grad()
def inference_ensemble(models, loader, device):
    results = {}
    for m_idx, model in enumerate(models):
        for images, names in tqdm(loader, desc=f"Model {m_idx+1}/{len(models)}"):
            images = images.to(device)
            probs = model.predict_proba(images).cpu().numpy()
            for name, prob in zip(names, probs):
                results.setdefault(name, []).append(float(prob))
    return [{"image_name": k, "score": sum(v) / len(v)} for k, v in results.items()]


class PatchDataset(Dataset):
    """Generates multiple crops per image for patch-based inference."""
    def __init__(self, base_dataset, image_size=224, n_random=3, backbone_type="clip"):
        self.base = base_dataset
        self.size = image_size
        self.n_random = n_random
        mean, std = _get_norm(backbone_type)
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.base)

    def _get_image_path(self, idx):
        if hasattr(self.base, 'samples'):
            s = self.base.samples[idx]
            return s[0] if isinstance(s, (tuple, list)) else s
        if hasattr(self.base, 'dataset') and hasattr(self.base.dataset, 'samples'):
            s = self.base.dataset.samples[self.base.indices[idx]]
            return s[0] if isinstance(s, (tuple, list)) else s
        raise RuntimeError("Cannot resolve image path from dataset")

    def _crop(self, img, box):
        return img.crop(box).resize((self.size, self.size), Image.BILINEAR)

    def __getitem__(self, idx):
        name = self.base[idx][1]
        img = Image.open(self._get_image_path(idx)).convert("RGB")
        w, h = img.size
        cs = max(min(w, h), 2)
        cx, cy = w // 2, h // 2
        half = cs // 2
        crops = [self._crop(img, (max(0, cx - half), max(0, cy - half),
                                   min(w, cx + half), min(h, cy + half)))]
        for x0, y0 in [(0, 0), (max(0, w - cs), 0), (0, max(0, h - cs)), (max(0, w - cs), max(0, h - cs))]:
            crops.append(self._crop(img, (x0, y0, min(w, x0 + cs), min(h, y0 + cs))))
        import random
        for _ in range(self.n_random):
            scale = random.uniform(0.6, 1.0)
            crop_size = max(int(cs * scale), 2)
            x0 = random.randint(0, max(0, w - crop_size))
            y0 = random.randint(0, max(0, h - crop_size))
            crops.append(self._crop(img, (x0, y0, min(w, x0 + crop_size), min(h, y0 + crop_size))))
        tensors = torch.stack([self.to_tensor(c) for c in crops])
        return tensors, name


@torch.no_grad()
def inference_patch(model, dataset, device, image_size=224,
                    n_random=3, backbone_type="clip"):
    patch_ds = PatchDataset(dataset, image_size, n_random, backbone_type)
    results = []
    for tensors, name in tqdm(patch_ds, desc="Patch inference"):
        tensors = tensors.to(device)
        probs = model.predict_proba(tensors).cpu().numpy()
        results.append({"image_name": name, "score": float(probs.mean())})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data")
    parser.add_argument("--val_dir", type=str, default=None, help="Validation image directory (flat layout)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--model_type", type=str, default="rine", choices=["timm", "rine", "dinov2", "dinov3", "dinov3_moe", "freq"])
    parser.add_argument("--backbone", type=str, default="ViT-L/14")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="/workspace/submissions/submission.csv")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--model_paths", type=str, default="", help="Comma-separated checkpoint paths for ensemble")
    parser.add_argument("--num_hooks", type=int, default=None)
    parser.add_argument("--lora_layers", type=int, default=0)
    parser.add_argument("--patch", action="store_true", help="Multi-crop patch inference")
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--n_random_crops", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_type = "clip" if args.model_type == "rine" else "imagenet"
    transform = get_val_transform(backbone_type=backbone_type)

    if args.toy:
        dataset = ToyDataset(args.data_root, transform=transform)
    elif args.val_dir:
        dataset = AIGIDataset(args.data_root, transform=transform, val_dir=args.val_dir)
    else:
        dataset = AIGIDataset(args.data_root, transform=transform, is_test=True)

    if args.ensemble and args.model_paths:
        paths = [p.strip() for p in args.model_paths.split(",")]
        models = [load_model(args.model_type, args.model, args.backbone, p, device, args.num_hooks, args.lora_layers, getattr(args, 'num_experts', 3)) for p in paths]
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        results = inference_ensemble(models, loader, device)
    elif args.patch:
        model = load_model(args.model_type, args.model, args.backbone, args.checkpoint, device, args.num_hooks, args.lora_layers, getattr(args, 'num_experts', 3))
        results = inference_patch(model, dataset, device, n_random=args.n_random_crops,
                                  backbone_type=backbone_type)
    elif args.tta:
        model = load_model(args.model_type, args.model, args.backbone, args.checkpoint, device, args.num_hooks, args.lora_layers, getattr(args, 'num_experts', 3))
        tta_tfms = get_tta_transforms(backbone_type=backbone_type)
        results = inference_tta(model, dataset, tta_tfms, args.batch_size, device)
    else:
        model = load_model(args.model_type, args.model, args.backbone, args.checkpoint, device, args.num_hooks, args.lora_layers, getattr(args, 'num_experts', 3))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        results = inference(model, loader, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
