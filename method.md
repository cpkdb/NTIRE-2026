# DINOv3-FreqMoE: AI-Generated Image Detection

## 1. 任务概述

NTIRE 2026 AI-Generated Image Detection 赛道。输入一张图片，输出 `score ∈ [0,1]`（1 = AI 生成）。评估指标为 AUC-ROC，重点考察对 JPEG 压缩、缩放、模糊、截图等后处理干扰下的鲁棒性。

## 2. 核心架构：DINOv3FreqMoE

```
Input Image (224×224)
    │
    ├──► [DINOv3 Backbone (frozen)] ──► Hook CLS tokens (norm2, last-N layers)
    │         │
    │         └──► LoRA (q_proj, v_proj, rank=8, last-K layers)
    │
    └──► [Haar DWT] ──► freq_enc (9ch → 128-d) ──► f_tok
              │
              ├──► alpha_gen(f_tok) → 动态 hook 聚合权重
              │         │
              │         └──► z = softmax(α) · proj1(hooks) → proj2(z)
              │
              ├──► router(f_tok) / τ → soft gates [B, E]
              │
              └──► experts[0..2](z) → expert_logits [B, E]
                        │
                        └──► output = Σ(gates · expert_logits)
```

### 2.1 组件说明

| 组件 | 结构 | 作用 |
|------|------|------|
| DINOv3 Backbone | `AutoModel` (hidden_size=1024), 全程 frozen | 提取多层语义特征 |
| LoRA | `LoRALinear(rank=8)` 注入 attention 的 q_proj/v_proj | 低秩适配，不改变 backbone 权重 |
| Hook 聚合 | 取最后 N 层 `norm2` 输出的 CLS token `[B, N, 1024]` | 多尺度特征 |
| proj1 | `Dropout → (Linear→ReLU→Dropout) × 3`，1024→256 | 降维 |
| Haar DWT | 固定 Haar 小波核，stride=2 分组卷积，提取 LH/HL/HH 共 9 通道 | 频域特征 |
| freq_enc | 4 层 Conv2d (9→32→64→96→128) + BN + ReLU + AdaptiveAvgPool | 频域编码为 128-d `f_tok` |
| alpha_gen | `Linear(128, N×256)` → softmax → 加权聚合 hook 特征 | 频率引导的动态 hook 注意力 |
| proj2 | `Dropout → (Linear→ReLU→Dropout) × 3`，256→256 | 特征精炼 |
| Router | `Linear(128,64)→ReLU→Linear(64,E)`, zero-init, 温度 τ (learnable) | 频率门控软路由 |
| Experts | 3 个独立 MLP: `Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear(1)` | 多专家分类头 |

### 2.2 关键设计

- **频率门控路由**：router 的输入是 `f_tok`（纯频域信息），而非语义特征。不同后处理方式（JPEG、模糊、缩放）在频域有显著差异，router 据此分配专家权重。
- **动态 alpha**：`alpha_gen` 根据频域 token 动态生成 hook 聚合权重，替代静态可学习 `alpha` 参数，使模型能根据输入图片的频率特性自适应选择关注哪些 backbone 层。
- **温度参数 τ**：可学习标量，clamp 在 `[0.1, 10.0]`，控制 router softmax 的锐度。
- **Zero-init router**：router 最后一层权重和偏置初始化为 0，训练初期 gates ≈ uniform(1/E)，等价于 ensemble，避免早期路由坍缩。

## 3. 四阶段渐进式训练

整体思路：先分别训练语义分支和频域分支，再通过 warm-start + expert mutation 组装 MoE，分两阶段微调。

### Stage 1: DINOv3 + LoRA 基础训练

```bash
# 训练 DINOv3Classifier (单头)
--model_type dinov3 --lora_layers 6 --epochs 15
```

- 冻结 backbone，仅训练 LoRA + proj1 + proj2 + head + alpha
- 输出：`dinov3_v2_lora/best.pt`（含 LoRA 权重、proj、head、alpha）

### Stage 2: FreqClassifier 预训练

```bash
# 独立训练频域分类器
--model_type freq --epochs 15
```

- 训练 Haar DWT → enc → fc 的完整频域分类器
- 输出：`freq_v2/best.pt`（含 enc 权重）

### Stage 3 (Stage A): MoE 组装 + 冻结 LoRA 微调

```bash
--model_type dinov3_moe --num_experts 3 --lora_layers 6 \
  --warm_start dinov3_v2_lora/best.pt \
  --freq_ckpt freq_v2/best.pt \
  --freeze_lora_only --moe_lr_scale 5.0 --epochs 3
```

**Warm-start 流程**：
1. `load_trainable(dinov3_v2_lora/best.pt)`：加载 LoRA + proj 权重；检测到 `head.*` 键但无 `experts.*` 键，触发 Expert Mutation：
   - Expert 0 = head 权重原样复制（clean anchor）
   - Expert 1, 2 = head 权重 + `N(0, 1e-3)` 高斯噪声（打破对称性）
   - 移除旧 `head.*` 和 `alpha` 键
2. `load_freq_encoder(freq_v2/best.pt)`：将 FreqClassifier 的 `enc.*` 映射到 `freq_enc.*`

**训练配置**：
- 冻结 LoRA 参数（`freeze_lora_only`）
- 3 组学习率：`moe_new`（router/alpha_gen/freq_enc/τ）× 5.0，`base`（proj/experts）× 1.0，`lora` 冻结
- 训练新增组件（router、alpha_gen）与已有组件（proj、experts）的协同

### Stage 4 (Stage B): 全参数微调

```bash
--model_type dinov3_moe --num_experts 3 --lora_layers 6 \
  --resume best_stage_a.pt \
  --lora_lr_scale 0.1 --moe_lr_scale 3.0 --epochs 3
```

- 解冻 LoRA，全部参数可训练
- LoRA 学习率 × 0.1（防止破坏已收敛的适配）
- MoE 新组件学习率 × 3.0（继续优化路由）

## 4. 损失函数

```
L = L_cls + L_contrastive + λ_lb · L_balance + λ_cons · L_consistency
```

| 损失 | 公式 | 作用 |
|------|------|------|
| L_cls | `BCEWithLogitsLoss` + label smoothing (ε=0.05) | 主分类损失 |
| L_contrastive | 基于 cosine similarity 的正负对对比损失 | 拉近同类、推远异类的特征空间 |
| L_balance | `E · Σ(avg_gate²)`，线性衰减 | 防止 router 坍缩到单一专家 |
| L_consistency | `BCE(σ(corrupt_logits), σ(clean_logits).detach())` | 对同一图片的 clean/corrupt 视图输出一致性约束 |

- Consistency loss 有 ramp-up 机制（前 `cons_ramp_epochs` 个 epoch 线性增长）
- Load balance loss 随训练线性衰减（后期允许专家特化）

## 5. 数据增强

### 5.1 训练增强 (RobustTransform)

40% 概率 light path，60% 概率 strong path：

- **Light**: Resize(224) → HFlip
- **Strong**: Resize(224) → HFlip → ColorJitter → GaussianBlur → Downsample-Upsample → Sharpen → MedianFilter → GammaAdjust → JPEG/WebP/DoubleJPEG → LensBlur → ColorShift → ImpulseNoise → SpatialJitter → ColorQuantization → ScreenshotSim → Perspective → CompoundCorruption → GaussianNoise → RandomErasing

### 5.2 Consistency 增强

对同一图片生成 (clean, corrupt) 对：
- Clean: Resize + 可选 HFlip + Normalize
- Corrupt: 在 clean 基础上随机叠加 1-3 种退化（GaussianBlur / JPEG(5-60) / Downsample / MedianFilter / Brightness / DoubleJPEG / Screenshot / LensBlur / ColorShift / ImpulseNoise / SpatialJitter / ColorQuantization）

### 5.3 Robust 验证增强

验证时随机施加一种退化（JPEG(30-50) / GaussianBlur / Downsample / MedianFilter / CompoundCorruption / LensBlur / ImpulseNoise / SpatialJitter / ColorQuantization）+ GaussianNoise，用于估计鲁棒 AUC。

## 6. 推理

支持 4 种模式：
- **Standard**: Resize(224) → Normalize → forward → sigmoid
- **TTA**: 5 视图平均（原图 / HFlip / VFlip / JPEG(q=75) / ScaleCrop(1.1×)）
- **Patch**: 中心裁剪 + 4 角裁剪 + N 随机裁剪，取均值
- **Ensemble**: 多 checkpoint 概率平均

## 7. 实验结果

| 版本 | 方法 | Local Robust AUC | Codabench |
|------|------|-----------------|-----------|
| v1 | DINOv3 + LoRA (单头) | 0.9943 | ~0.90 |
| v3 Stage A | DINOv3-FreqMoE (freeze LoRA) | 0.9934 | - |
| v3 Stage B | DINOv3-FreqMoE (full finetune) | **0.9952** | ~0.90 |

v3 在 local robust AUC 上有提升，但 Codabench 提交分数与 v1 持平（~0.90），说明 local robust validation 与线上评估存在 distribution gap。

## 8. 文件结构

```
src/
├── models/
│   ├── dinov3_wrapper.py      # DINOv3Classifier (backbone + LoRA + hook + proj + head)
│   ├── dinov3_freq_moe.py     # DINOv3FreqMoE (继承 DINOv3Classifier, 加 freq_enc + router + experts)
│   ├── freq_classifier.py     # FreqClassifier (独立频域分类器, Stage 2 预训练用)
│   └── ...
├── transforms.py              # 所有数据增强 (RobustTransform, ConsistencyTransform, TTA, etc.)
├── train.py                   # 训练入口 (4 阶段统一, 参数分组, 冻结控制)
└── inference.py               # 推理入口 (standard / TTA / patch / ensemble)
```
