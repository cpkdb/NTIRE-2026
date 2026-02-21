# NTIRE 2026 AI 生成图像检测 - 方法文档

## 整体方案

双 Backbone 集成：**CLIP ViT-L/14** + **DINOv2-L/14-reg**，共享 RINE 架构，独立训练，推理时加权融合。

```
                    ┌─────────────────────┐
                    │     输入图像 224×224  │
                    └──────┬──────┬───────┘
                           │      │
              ┌────────────▼┐  ┌──▼────────────┐
              │  Model A     │  │  Model B       │
              │  CLIP-RINE   │  │  DINOv2-RINE   │
              │  (v4)        │  │  (dinov2_v1)   │
              └──────┬───────┘  └──────┬─────────┘
                     │ TTA×5          │ TTA×5
                     │ score_clip     │ score_dino
                     └──────┬──┬──────┘
                            │  │
                     加权平均 (w_clip : w_dino)
                            │
                         最终 score
```

---

## 一、共享训练框架

### 1.1 数据
- **训练集**: shard_0 + shard_1，共 100,000 张（50% Real / 50% AI）
- **验证**: 从训练集随机划出 10%（10,000 张）
- **验证策略**: 同时评估 Clean Val AUC 和 Robust Val AUC，按 **Robust Val AUC** 选择最佳 checkpoint

### 1.2 数据增强 (RobustTransform)

两条路径随机切换（Light 30% / Strong 70%）：

**Light 路径**: Resize(224) → RandomHFlip → ToTensor → Normalize

**Strong 路径**:
```
Resize(224) → RandomHFlip
  → ColorJitter(0.3, 0.3, 0.3, 0.1)  p=0.4
  → GaussianBlur(k=5, σ=0.1-3.0)     p=0.3
  → RandomDownsampleUpsample(0.25-1.0)
  → RandomSharpen                      p=0.3
  → MedianFilter(k=3)                  p=0.2
  → GammaAdjust(0.7-1.3)              p=0.3
  → RandomChoice[JPEG(q=10-100), WebP(q=10-100)]
  → ToTensor
  → GaussianNoise(std=0-0.08)
  → Normalize
  → RandomErasing                      p=0.1
```

### 1.3 损失函数（3 组件）
```
L = L_bce(平滑标签) + 1.0 × L_contrastive + 0.5 × L_consistency
```

- **标签平滑**: 0/1 → 0.025/0.975（ε=0.05）
- **对比损失**: 特征空间内正样本对拉近、负样本对推远
- **一致性损失**: ConsistencyDataset 生成 (干净视图, 扰动视图) 配对，模型在扰动视图上的 sigmoid 输出对齐干净视图（干净侧梯度截断），扰动操作从 {高斯模糊, JPEG(20-60), 下采样(0.3-0.7), 中值滤波, 亮度调节} 中随机组合 1-3 种

### 1.4 训练配置
| 参数 | 值 |
|------|-----|
| Epochs | 15 |
| Batch size | 32 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=15) |
| num_hooks | 12（仅使用 backbone 最后 12 层） |

### 1.5 Robust 验证集变换
随机施加一种扰动：JPEG(q=30-50) / GaussianBlur(σ=1-2) / 下采样(0.4-0.6) / MedianFilter(k=3)，加轻度噪声(std=0.01-0.03)。

---

## 二、Model A — CLIP-RINE (v4)

### 2.1 Backbone
- **CLIP ViT-L/14**（冻结），24 Transformer Blocks
- 归一化: CLIP 均值/标准差 `(0.481, 0.458, 0.408) / (0.269, 0.261, 0.276)`

### 2.2 特征提取
- Hook 点: 各 Block 的 `ln_2`（FFN 后 LayerNorm）
- 使用后 12 层，取每层全 patch 输出 → [B, N_patches, 1024] × 12

### 2.3 可训练头部 (RINE)
```
ln_2 输出 [B, N_patches, 1024] × 12
  → proj1: Dropout → (Linear(1024→256) → ReLU → Dropout) × 3
  → alpha: softmax 加权求和（跨 12 层）→ [B, N_patches, 256]
  → 对 patches 维度求和 → [B, 256]
  → proj2: Dropout → (Linear(256→256) → ReLU → Dropout) × 3
  → head: Linear(256→256) → ReLU → Dropout → Linear(256→256) → ReLU → Dropout → Linear(256→1)
```

### 2.4 训练结果
- 本地 Robust Val AUC: **0.9761** (Epoch 14/15)
- 排行榜单模型: Clean AUC 0.9727, Robust AUC **0.9134**

---

## 三、Model B — DINOv2-RINE (dinov2_v1)

### 3.1 Backbone
- **DINOv2-L/14-reg**（冻结，自监督预训练，304M 参数）
- 24 Transformer Blocks，embed_dim=1024，4 register tokens
- 归一化: ImageNet 均值/标准差 `(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)`

### 3.2 特征提取
- Hook 点: 各 Block 的 `norm2`（MLP 后 LayerNorm）
- 使用后 12 层，取每层 **CLS token** → [B, 1024] × 12

### 3.3 可训练头部
```
norm2 CLS 输出 [B, 1024] × 12 → stack → [B, 12, 1024]
  → proj1: Dropout → (Linear(1024→256) → ReLU → Dropout) × 3 → [B, 12, 256]
  → alpha: softmax 加权求和（跨 12 层）→ [B, 256]
  → proj2: Dropout → (Linear(256→256) → ReLU → Dropout) × 3
  → head: Linear(256→256) → ReLU → Dropout → Linear(256→256) → ReLU → Dropout → Linear(256→1)
```
可训练参数: 726K，GPU 占用 ~2.6 GB (RTX 4090)

### 3.4 与 CLIP-RINE 的关键区别
| | CLIP-RINE | DINOv2-RINE |
|---|---|---|
| 预训练方式 | 语言-图像对比学习 | 自监督蒸馏 |
| 特征偏向 | 语义级别 | 像素级结构一致性 |
| Hook 层 | `ln_2` | `norm2` |
| 聚合对象 | 全 patch 求和 | 仅 CLS token |
| 归一化 | CLIP 专用 | ImageNet 标准 |

### 3.5 训练结果
```
Epoch  1: Robust 0.9486
Epoch  5: Robust 0.9674
Epoch 10: Robust 0.9751
Epoch 14: Robust 0.9771 ← best
Epoch 15: Robust 0.9760
```
- 本地 Robust Val AUC: **0.9771** (Epoch 14/15)
- 排行榜单模型 (TTA): Robust AUC **0.9339**

---

## 四、推理策略

### 4.1 TTA (Test-Time Augmentation)
每个模型 5 路 TTA，对预测概率取均值：
1. 原图
2. 水平翻转
3. 垂直翻转
4. JPEG 压缩 (q=75)
5. 缩放裁剪 (1.1× → CenterCrop)

### 4.2 双模型加权 Ensemble
```python
score = w_clip × score_clip_tta + w_dino × score_dino_tta
```
- 等权 (0.5 / 0.5): Robust AUC **0.9414**
- 加权实验中（DINOv2 权重更高，因其单模型表现更强）

### 4.3 已验证无效的方案
- **Patch 推理**（多裁剪局部分析）: 随机裁剪在排行榜扰动下稀释信号，等权 ensemble 0.9414 → 加入 patch 后 0.9382，有害

---

## 五、排行榜结果

| 版本 | 方案 | Robust AUC |
|------|------|-----------|
| v3 | CLIP-RINE, 50k 数据 | 0.9134 |
| v4 | CLIP-RINE, 100k + 一致性训练 | 0.9134 |
| v5 | DINOv2-RINE 单模型 + TTA | 0.9339 |
| v6 | CLIP + DINOv2 等权 ensemble | **0.9414** |
| v6+ | 加权 ensemble | 实验中 |

---

## 六、版本演进

### v1（基线）
- CLIP-RINE, shard_0 (50k), 基础增强, 仅 BCE 损失
- Robust AUC ~0.85

### v2（+ 鲁棒增强 + 对比损失）
- RobustTransform, 3路TTA
- Robust AUC ~0.88

### v3（+ 更强增强）
- 增强升级 (WebP, 更宽参数范围), 4路TTA
- 排行榜 Robust AUC **0.9134**

### v4（+ 一致性训练 + 数据扩充）
- 100k 数据, 一致性损失, 标签平滑, num_hooks=12
- 本地 Robust Val AUC 0.9761，排行榜 Robust AUC 0.9134（与 v3 持平）

### v5（DINOv2 Backbone）
- DINOv2-L/14-reg 替换 CLIP，同样训练框架
- 排行榜单模型 Robust AUC **0.9339** (+0.0205)

### v6（双模型 Ensemble）
- CLIP-RINE TTA + DINOv2-RINE TTA 分数融合
- 排行榜 Robust AUC **0.9414** (+0.0075)

---

## 七、关键发现

1. **DINOv2 > CLIP 在鲁棒性上**: DINOv2 自监督学习的结构特征比 CLIP 语义特征对后处理更鲁棒（排行榜 0.9339 vs 0.9134）
2. **Ensemble 互补有效**: 两个 backbone 的错误模式不同，融合后稳定提升
3. **本地-排行榜差距**: 本地 Robust Val AUC ~0.977 vs 排行榜 ~0.94，说明排行榜扰动分布更极端
4. **一致性训练**: 显式强制模型在干净/扰动输入上预测一致，直接优化鲁棒性目标
5. **Patch 推理无效**: 随机裁剪在极端扰动下反而稀释判别信号
6. **层选择**: 仅用后 12 层 Hook，丢弃低层脆弱像素特征，起到正则化效果

---

## 八、文件结构
```
src/
├── train.py              # 训练循环（一致性、对比、标签平滑、双验证）
├── inference.py           # 推理（TTA、Patch、Ensemble）
├── transforms.py          # 增强（RobustTransform、ConsistencyTransform、TTA、Stress-suite）
├── datasets/
│   └── aigi_dataset.py    # 数据集加载（shard 0-5、val_dir）
├── models/
│   ├── __init__.py
│   ├── base_classifier.py
│   ├── rine_wrapper.py    # CLIP-RINE 封装
│   └── dinov2_wrapper.py  # DINOv2-RINE 封装
└── utils/
    └── metrics.py

baselines/rine/src/
└── models.py              # RINE 核心（Hooks、proj、alpha、head）
```

## 九、运行命令

```bash
# 训练 CLIP-RINE
python src/train.py --data_root /data --model_type rine --backbone "ViT-L/14" \
    --epochs 15 --batch_size 32 --lr 1e-4 \
    --consistency_weight 0.5 --label_smoothing 0.05 --num_hooks 12

# 训练 DINOv2-RINE
python src/train.py --data_root /data --model_type dinov2 \
    --epochs 15 --batch_size 32 --lr 1e-4 \
    --consistency_weight 0.5 --label_smoothing 0.05 --num_hooks 12

# 推理（TTA）
python src/inference.py --val_dir /path/to/val --model_type rine \
    --checkpoint best.pt --backbone "ViT-L/14" --num_hooks 12 --tta

python src/inference.py --val_dir /path/to/val --model_type dinov2 \
    --checkpoint best.pt --num_hooks 12 --tta

# Ensemble（Python 脚本）
# score = w_clip × clip_score + w_dino × dino_score
```
