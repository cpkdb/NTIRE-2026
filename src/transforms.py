import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import convolve, map_coordinates
import torch
import torchvision.transforms as T
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORM_PRESETS = {
    "clip": (CLIP_MEAN, CLIP_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
}


def _get_norm(backbone_type="clip"):
    mean, std = NORM_PRESETS.get(backbone_type, NORM_PRESETS["clip"])
    return mean, std


class JPEGCompression:
    def __init__(self, quality_range=(10, 100)):
        self.quality_range = quality_range

    def __call__(self, img):
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class WebPCompression:
    def __init__(self, quality_range=(10, 100)):
        self.quality_range = quality_range

    def __call__(self, img):
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    def __init__(self, std_range=(0.0, 0.05)):
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        return torch.clamp(tensor + torch.randn_like(tensor) * std, 0, 1)


class RandomDownsampleUpsample:
    def __init__(self, scale_range=(0.25, 1.0)):
        self.scale_range = scale_range

    def __call__(self, img):
        s = random.uniform(*self.scale_range)
        if s >= 0.99:
            return img
        w, h = img.size
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
        return img.resize((w, h), Image.BILINEAR)


class RandomSharpen:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.SHARPEN)
        return img


class MedianFilter:
    def __init__(self, kernel_size=3, p=0.2):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.MedianFilter(size=self.kernel_size))
        return img


class GammaAdjust:
    def __init__(self, gamma_range=(0.7, 1.3), p=0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return ImageEnhance.Brightness(img).enhance(gamma)
        return img


class LensBlur:
    def __init__(self, radius_range=(1, 5)):
        self.radius_range = radius_range

    def __call__(self, img):
        radius = random.randint(*self.radius_range)
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel = ((x ** 2 + y ** 2) <= radius ** 2).astype(np.float32)
        kernel /= kernel.sum()
        arr = np.array(img, dtype=np.float32)
        for c in range(3):
            arr[:, :, c] = convolve(arr[:, :, c], kernel, mode='reflect')
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class ColorShift:
    def __init__(self, amount_range=(1, 8)):
        self.amount_range = amount_range

    def __call__(self, img):
        amount = random.randint(*self.amount_range)
        arr = np.array(img, dtype=np.float32)
        channel = random.randint(0, 2)
        angle = random.uniform(0, 2 * np.pi)
        dx = int(round(np.cos(angle) * amount))
        dy = int(round(np.sin(angle) * amount))
        arr[:, :, channel] = np.roll(np.roll(arr[:, :, channel], dx, axis=1), dy, axis=0)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class ImpulseNoise:
    def __init__(self, density_range=(0.001, 0.02)):
        self.density_range = density_range

    def __call__(self, img):
        arr = np.array(img)
        d = random.uniform(*self.density_range)
        n_pixels = int(d * arr.shape[0] * arr.shape[1])
        n_salt = n_pixels // 2
        coords = (np.random.randint(0, arr.shape[0], n_salt),
                  np.random.randint(0, arr.shape[1], n_salt))
        arr[coords] = 255
        n_pepper = n_pixels - n_salt
        coords = (np.random.randint(0, arr.shape[0], n_pepper),
                  np.random.randint(0, arr.shape[1], n_pepper))
        arr[coords] = 0
        return Image.fromarray(arr)


class SpatialJitter:
    def __init__(self, amount_range=(0.1, 0.5)):
        self.amount_range = amount_range

    def __call__(self, img):
        amount = random.uniform(*self.amount_range)
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        dy = np.random.randn(h, w) * amount
        dx = np.random.randn(h, w) * amount
        y, x = np.mgrid[0:h, 0:w]
        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = map_coordinates(arr[:, :, c], [y + dy, x + dx], order=1, mode='reflect')
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


class ColorQuantization:
    def __init__(self, levels_range=(7, 20)):
        self.levels_range = levels_range

    def __call__(self, img):
        levels = random.randint(*self.levels_range)
        arr = np.array(img, dtype=np.float32)
        step = 255.0 / levels
        arr = (np.floor(arr / step) * step).clip(0, 255)
        return Image.fromarray(arr.astype(np.uint8))


class CompoundCorruption:
    def __init__(self, quality_range=(10, 60)):
        self.ops = [
            JPEGCompression(quality_range),
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))),
            RandomDownsampleUpsample(scale_range=(0.3, 0.8)),
            RandomSharpen(p=1.0),
            MedianFilter(kernel_size=3, p=1.0),
            LensBlur(radius_range=(1, 4)),
            ColorShift(amount_range=(1, 6)),
            ImpulseNoise(density_range=(0.001, 0.015)),
            SpatialJitter(amount_range=(0.1, 0.4)),
            ColorQuantization(levels_range=(8, 16)),
        ]

    def __call__(self, img):
        for op in random.sample(self.ops, random.randint(2, 3)):
            img = op(img)
        return img


class RobustTransform:
    def __init__(self, image_size=224, backbone_type="clip"):
        mean, std = _get_norm(backbone_type)
        self.light = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.strong = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.4),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.3),
            RandomDownsampleUpsample(scale_range=(0.25, 1.0)),
            RandomSharpen(p=0.3),
            MedianFilter(kernel_size=3, p=0.2),
            GammaAdjust(gamma_range=(0.7, 1.3), p=0.3),
            T.RandomChoice([
                JPEGCompression(quality_range=(10, 100)),
                WebPCompression(quality_range=(10, 100)),
                DoubleJPEG(),
            ]),
            T.RandomApply([LensBlur(radius_range=(1, 4))], p=0.15),
            T.RandomApply([ColorShift(amount_range=(1, 6))], p=0.15),
            T.RandomApply([ImpulseNoise(density_range=(0.001, 0.015))], p=0.1),
            T.RandomApply([SpatialJitter(amount_range=(0.1, 0.4))], p=0.1),
            T.RandomApply([ColorQuantization(levels_range=(8, 18))], p=0.1),
            T.RandomApply([ScreenshotSim()], p=0.15),
            T.RandomPerspective(distortion_scale=0.2, p=0.15),
            T.RandomApply([CompoundCorruption()], p=0.2),
            T.ToTensor(),
            GaussianNoise(std_range=(0.0, 0.08)),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])

    def __call__(self, img):
        return self.light(img) if random.random() < 0.3 else self.strong(img)


# Corruption function for consistency training (applied to tensors after base transform)
def random_corrupt_pil(img):
    ops = [
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0))),
        lambda x: JPEGCompression(quality_range=(5, 60))(x),
        lambda x: RandomDownsampleUpsample(scale_range=(0.3, 0.7))(x),
        lambda x: x.filter(ImageFilter.MedianFilter(size=3)),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: DoubleJPEG()(x),
        lambda x: ScreenshotSim()(x),
        lambda x: LensBlur(radius_range=(1, 4))(x),
        lambda x: ColorShift(amount_range=(1, 6))(x),
        lambda x: ImpulseNoise(density_range=(0.001, 0.015))(x),
        lambda x: SpatialJitter(amount_range=(0.1, 0.4))(x),
        lambda x: ColorQuantization(levels_range=(8, 16))(x),
    ]
    n = random.randint(1, 3)
    chosen = random.sample(ops, min(n, len(ops)))
    for op in chosen:
        img = op(img)
    return img


class ConsistencyTransform:
    """Returns (clean_view, corrupted_view) for consistency training."""
    def __init__(self, image_size=224, backbone_type="clip"):
        self.size = image_size
        mean, std = _get_norm(backbone_type)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.BILINEAR)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        clean = self.to_tensor(img)
        corrupted_img = random_corrupt_pil(img)
        corrupted = self.to_tensor(corrupted_img)
        return clean, corrupted


def get_train_transform(image_size=224, robust=True, backbone_type="clip"):
    if robust:
        return RobustTransform(image_size, backbone_type=backbone_type)
    mean, std = _get_norm(backbone_type)
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_consistency_transform(image_size=224, backbone_type="clip"):
    return ConsistencyTransform(image_size, backbone_type=backbone_type)


def get_val_transform(image_size=224, backbone_type="clip"):
    mean, std = _get_norm(backbone_type)
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_robust_val_transform(image_size=224, backbone_type="clip"):
    """Val transform with perturbations for robust AUC estimation."""
    mean, std = _get_norm(backbone_type)
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomChoice([
            JPEGCompression(quality_range=(30, 50)),
            T.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0)),
            RandomDownsampleUpsample(scale_range=(0.4, 0.6)),
            MedianFilter(kernel_size=3, p=1.0),
            CompoundCorruption(quality_range=(20, 40)),
            LensBlur(radius_range=(2, 5)),
            ImpulseNoise(density_range=(0.005, 0.02)),
            SpatialJitter(amount_range=(0.2, 0.5)),
            ColorQuantization(levels_range=(7, 13)),
        ]),
        T.ToTensor(),
        GaussianNoise(std_range=(0.01, 0.03)),
        T.Normalize(mean=mean, std=std),
    ])


def get_tta_transforms(image_size=224, backbone_type="clip"):
    mean, std = _get_norm(backbone_type)
    base = get_val_transform(image_size, backbone_type)
    hflip = T.Compose([
        T.Resize((image_size, image_size)),
        T.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    vflip = T.Compose([
        T.Resize((image_size, image_size)),
        T.Lambda(lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    jpeg_tta = T.Compose([
        T.Resize((image_size, image_size)),
        JPEGCompression(quality_range=(75, 75)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    scale_crop = T.Compose([
        T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return [base, hflip, vflip, jpeg_tta, scale_crop]


class DoubleJPEG:
    def __init__(self, q1_range=(30, 70), q2_range=(50, 90)):
        self.q1_range = q1_range
        self.q2_range = q2_range

    def __call__(self, img):
        img = JPEGCompression(self.q1_range)(img)
        return JPEGCompression(self.q2_range)(img)


class ScreenshotSim:
    """Simulate screenshot pipeline: resize -> JPEG -> slight blur -> JPEG."""
    def __call__(self, img):
        w, h = img.size
        s = random.uniform(0.5, 0.8)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        img = JPEGCompression((40, 70))(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        return JPEGCompression((60, 85))(img)


def get_stress_val_transforms(image_size=224, backbone_type="clip"):
    """Returns dict of {corruption_name: transform} for per-corruption eval."""
    mean, std = _get_norm(backbone_type)

    def _wrap(corrupt_fn):
        return T.Compose([
            T.Resize((image_size, image_size)),
            corrupt_fn,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return {
        "double_jpeg": _wrap(DoubleJPEG()),
        "heavy_jpeg": _wrap(JPEGCompression((10, 25))),
        "webp_low": _wrap(WebPCompression((10, 30))),
        "screenshot": _wrap(ScreenshotSim()),
        "heavy_blur": _wrap(T.GaussianBlur(kernel_size=7, sigma=(2.0, 4.0))),
        "heavy_downsample": _wrap(RandomDownsampleUpsample((0.15, 0.35))),
        "median_filter": _wrap(MedianFilter(kernel_size=5, p=1.0)),
        "sharpen_strong": _wrap(RandomSharpen(p=1.0)),
        "gamma_extreme": _wrap(GammaAdjust(gamma_range=(0.5, 1.5), p=1.0)),
        "lens_blur": _wrap(LensBlur(radius_range=(3, 6))),
        "color_shift": _wrap(ColorShift(amount_range=(4, 10))),
        "impulse_noise": _wrap(ImpulseNoise(density_range=(0.01, 0.03))),
        "spatial_jitter": _wrap(SpatialJitter(amount_range=(0.3, 0.8))),
        "color_quantization": _wrap(ColorQuantization(levels_range=(7, 10))),
        "noise_heavy": T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            GaussianNoise(std_range=(0.05, 0.12)),
            T.Normalize(mean=mean, std=std),
        ]),
    }
