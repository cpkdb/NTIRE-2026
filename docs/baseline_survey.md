# AI-Generated Image Detection 论文调研报告 (更新版)

## 比赛核心要求分析

根据 NTIRE 2026 Challenge Overview：
1. **鲁棒性 (Robustness)**: 图像经过裁剪、缩放、压缩、模糊等 "in-the-wild" 变换后仍能检测
2. **泛化性 (Generalization)**: 能检测未见过的生成器产生的图像
3. **评估指标**: **Robust ROC AUC (主)** + Clean ROC AUC (次)

> ⚠️ **关键洞察**: 比赛主指标是 **Robust AUC**，意味着模型在各种图像变换后的表现比原始图像上的表现更重要。

---

## FatFormer vs RINE 详细对比

### RINE (ECCV 2024)

**论文**: [Leveraging Representations from Intermediate Encoder-Blocks for Synthetic Image Detection](https://arxiv.org/abs/2402.19091)

**GitHub**: https://github.com/mever-team/rine

**核心方法**:
- 利用 CLIP 中间层 Transformer blocks 的特征（而非仅最后一层）
- 轻量级网络将中间层表示映射到可学习的伪造感知向量空间
- 可训练的注意力模块整合各层重要性

**关键创新**:
- 中间层编码细粒度细节，比高层语义特征更适合检测伪造
- 浅层捕获低级视觉信息

**性能**:
| 指标 | 数值 |
|------|------|
| 测试数据集数量 | 20个 |
| 平均性能提升 | **+10.6%** (vs SOTA) |
| 训练时间 | **1 epoch (~8分钟)** |

---

### FatFormer (CVPR 2024)

**论文**: [Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection](https://arxiv.org/abs/2312.16649)

**GitHub**: https://github.com/Michel-liu/FatFormer

**核心方法**:
- Forgery-aware Adapter (FAA): 图像域 + 频率域特征提取
- 频率域使用 DWT (离散小波变换) + 分组注意力
- Language-guided Alignment (LGA): 文本引导的对齐增强泛化

**性能**:
| 数据集类型 | FatFormer | UnivFD (baseline) |
|------------|-----------|-------------------|
| GANs ACC | **98.4%** | 89.1% |
| GANs AP | **99.7%** | 98.3% |
| Diffusion ACC | **95.0%** | 85.4% |
| Diffusion AP | **98.8%** | 94.6% |

---

### 对比总结

| 维度 | RINE | FatFormer | 对比赛的影响 |
|------|------|-----------|-------------|
| **泛化性** | ✅ 强 (+10.6%) | ✅ 很强 (GAN→Diffusion) | 两者都好 |
| **鲁棒性** | ⚠️ 未明确报告 | ✅ 频率域特征抗压缩 | **FatFormer 更优** |
| **训练效率** | ✅ 1 epoch | ⚠️ 需要更多训练 | RINE 更快 |
| **参数量** | ✅ 轻量 | ⚠️ 493M | RINE 更轻 |
| **代码可用性** | ✅ 开源 | ✅ 开源 | 两者都可用 |
| **频率域特征** | ❌ 无 | ✅ DWT | **FatFormer 更优** |

---

## 针对 "In-the-Wild" 比赛的推荐

### 🥇 首选: FatFormer

**理由**:
1. **频率域特征是关键**: JPEG压缩、模糊等变换主要影响高频信息，FatFormer的DWT模块专门处理这类问题
2. **比赛主指标是 Robust AUC**: FatFormer 的设计更符合鲁棒性需求
3. **跨架构泛化**: 仅用 ProGAN 训练，能泛化到 Diffusion 模型

### 🥈 备选: RINE

**理由**:
1. **训练效率极高**: 1 epoch 即可收敛，适合快速迭代
2. **中间层特征**: 可能捕获到不同于最终层的伪造痕迹
3. **可与 FatFormer 集成**: 两种方法互补

### 🥉 组合策略 (推荐)

```
方案A: FatFormer 单模型 + 强数据增强
方案B: FatFormer + RINE 集成 (投票/加权平均)
方案C: 多模型集成 (FatFormer + RINE + DeeCLIP)
```

---

## 其他重要论文 (In-the-Wild 相关)

### 1. Navigating the Challenges of AI-Generated Image Detection in the Wild (2025)
- **关键发现**: 现有模型在受控数据集上表现好，但在真实场景下显著下降
- **四个关键因素**: backbone架构、训练数据组成、预处理策略、数据增强配置

### 2. A Simple yet Effective Framework for Blur-Robust AI-Generated Image Detection (2025)
- **方法**: Teacher-Student 知识蒸馏
- **Teacher**: DINOv3 (在清晰图像上训练)
- **适用场景**: 专门针对模糊鲁棒性

### 3. DeeCLIP (2025)
- **鲁棒性提升**: 71.91% vs SOTA 61.55% (**+10.36%**)
- **方法**: CLIP-ViT + LoRA + DeeFuser 多尺度融合

### 4. Deepfake-Eval-2024 Benchmark
- **发现**: SOTA 模型在真实社交媒体数据上 AUC 下降 **50%**
- **启示**: 需要更多真实场景数据增强

---

## 数据增强策略 (关键)

根据多篇论文的发现，**数据增强是提升鲁棒性的最有效手段**：

```python
import albumentations as A

robust_transforms = A.Compose([
    # JPEG 压缩 (最重要)
    A.ImageCompression(quality_lower=30, quality_upper=95, p=0.5),

    # 模糊
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 7)),
    ], p=0.3),

    # 缩放
    A.RandomScale(scale_limit=(-0.5, 0.5), p=0.3),

    # 裁剪
    A.RandomCrop(height=224, width=224, p=0.5),

    # 噪声
    A.GaussNoise(var_limit=(10, 50), p=0.2),

    # 颜色变换 (社交媒体常见)
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
])
```

---

## 实施路线图

```
Phase 1: 快速验证 (1-2天)
├── 搭建 FatFormer 环境
├── 在 Toy Dataset 上测试
└── 验证提交格式

Phase 2: 基线训练 (3-5天)
├── FatFormer 在完整数据集上训练
├── 加入数据增强
└── 评估 Clean/Robust AUC

Phase 3: 优化迭代
├── 尝试 RINE 作为补充
├── 模型集成实验
└── 超参数调优

Phase 4: 最终提交
├── 选择最佳模型/集成
├── 在官方镜像中验证
└── 生成 submission.csv
```

---

## 参考文献

1. Koutlis & Papadopoulos. "Leveraging Representations from Intermediate Encoder-Blocks for Synthetic Image Detection" **ECCV 2024**
2. Liu et al. "Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection" **CVPR 2024**
3. "A Robust and Generalizable Transformer-Based Framework for Detecting AI-Generated Images" 2025
4. "Navigating the Challenges of AI-Generated Image Detection in the Wild" 2025
5. "A Simple yet Effective Framework for Blur-Robust AI-Generated Image Detection" 2025
6. Wang et al. "CNN-generated images are surprisingly easy to spot... for now" CVPR 2020
7. Wang et al. "DIRE for Diffusion-Generated Image Detection" ICCV 2023
