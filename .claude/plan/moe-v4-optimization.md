# MoE v4.0 优化实施计划（修订版）

> 综合 Codex (SESSION: 019c9961-4db6-7393-8912-eba48d80dcff) + Gemini 分析 + 继承链断裂诊断

---

## 问题诊断总结

### P1: Signal Starvation（信号饥饿）
- **根因**：`f_tok` 仅来自 Haar DWT 高频子带 (LH/HL/HH)，重度模糊/下采样/JPEG 压缩下 HF 趋近零
- **后果**：router 在最难样本上退化为随机分配

### P2: Stage 2 目标冲突
- **根因**：FreqClassifier 用 BCE(real/fake) 预训练 → freq_enc 学到的是「真伪特征」而非「退化类型特征」
- **后果**：加载到 MoE 后 freq_enc 抗拒学习路由所需的退化诊断能力

### P3: Soft Routing 退化为 Ensemble
- **根因 1**：dynamic alpha + MoE soft gates 双重自适应，功能重叠
- **根因 2**：`effective_lb = lb_weight * max(0.2, ...)` 永远不归零，结构性阻止专家特化
- **后果**：gates ≈ uniform，三个专家学到几乎相同的映射

### P4: Expert 初始化对称性过强
- **根因**：mutation noise σ=1e-3（绝对值）不足以打破对称性
- **后果**：训练早期三个专家梯度近似，加剧 ensemble 退化

### P5: 继承链断裂（核心问题）
- **根因**：v3 用随机初始化的 `alpha_gen` 替换了 v1 学到的 `static alpha`，导致 `z` 的分布突变
- **后果**：`proj2` 和 `experts` 接收 OOD 输入，warm-start 名存实亡，前几个 epoch 本质是重新学习

---

## 设计原则：正交解耦

砍掉 `alpha_gen`，将架构解耦为两条**功能正交**的通路：

| 通路 | 组件 | 职责 | 来源 |
|------|------|------|------|
| **特征提取**（稳定） | hooks → proj1 → static alpha → proj2 → z | 在像素完好的前提下，榨干 DINOv3 的判别能力 | 从 v1 继承，严格保护 |
| **环境路由**（敏锐） | freq_enc(HF) + cls_router_proj(CLS.detach()) → router → gates | 全天候诊断退化类型，分配专家 | 全新组件，从零学习 |

两条通路的**唯一交汇点**是最终的加权求和：`output = Σ(gates · experts(z))`。

---

## v4 架构

```
Input Image (224×224)
    │
    ├──► [DINOv3 Backbone (frozen)] ──► Hook CLS tokens (norm2, last-N layers)
    │         │                              │
    │         └──► LoRA (q_proj, v_proj)     │
    │                                        │
    │    ┌───── 特征提取通路 (从 v1 继承) ──────┤
    │    │                                   │
    │    │  hooks → proj1 → g[B,N,256]       │
    │    │           ↓                       │
    │    │  static alpha [1,N,256]           │
    │    │           ↓                       │
    │    │  z = softmax(α)·g → sum → proj2   │
    │    │           ↓                       │
    │    │     experts[0..2](z) → logits     │
    │    │           ↓                       │
    │    └───────── [B, E] ──────────────┐   │
    │                                    │   │
    └──► [Haar DWT] ──► freq_enc ─► f_tok│   │
              │              [B,128]      │   │
              │                           │   │
              │  cls_router_proj(         │   │
              │    hooks[:,-1,:].detach()) │   │
              │         [B,128]           │   │
              │              │            │   │
              └── router_in = cat ────────│   │
                       [B,256]            │   │
                         ↓                │   │
                    router / τ            │   │
                         ↓                │   │
                    gates [B, E] ─────────┤
                                          ↓
                              output = Σ(gates · logits)
```

---

## 架构改动

### 改动 1：去掉 alpha_gen，保留 static alpha（解决 P5 + P3）

在 `dinov3_freq_moe.py` 的 `__init__` 中：
- **删除** `self.alpha_gen = nn.Linear(128, n_hooks * proj_dim)`
- **不再** `del self.alpha`，保留从父类继承的 `self.alpha`（static `[1, N, 256]`）

Forward 中使用 static alpha：
```python
g = self.proj1(self._extract(x).float())
z = (torch.softmax(self.alpha, dim=1) * g).sum(dim=1)  # 与 v1 完全一致
z = self.proj2(z)
```

### 改动 2：Hybrid Router Input（解决 P1）

新增 CLS 语义投影层（1024→128 对齐 f_tok 维度）：
```python
self.cls_router_proj = nn.Sequential(
    nn.LayerNorm(1024),
    nn.Linear(1024, 128),
    nn.GELU(),
    nn.Dropout(0.1),
)

# Router 输入维度 256 = 128(freq) + 128(semantic)
self.router = nn.Sequential(
    nn.Linear(256, 64), nn.ReLU(inplace=True),
    nn.Linear(64, num_experts),
)
```

Router 计算：
```python
cls_last = hooks[:, -1, :].detach()  # [B, 1024] - detach 隔离梯度
f_tok = self._freq_token(x)          # [B, 128] - 高频诊断
cls_tok = self.cls_router_proj(cls_last)  # [B, 128] - 低频/语义补盲
router_in = torch.cat([f_tok, cls_tok], dim=-1)  # [B, 256]
gates = softmax(self.router(router_in) / tau, dim=-1)
```

### 改动 3：Gumbel-Softmax 温度退火（解决 P3）

```python
self._use_gumbel = False
self._gumbel_tau = 1.0  # 退火: 1.0 → 0.1

# Forward 路由:
if self.training and self._use_gumbel:
    gates = F.gumbel_softmax(router_logits, tau=self._gumbel_tau, hard=True)
else:
    gates = softmax(router_logits, dim=-1)

# 推理 (eval): argmax 硬路由
if not self.training:
    indices = torch.argmax(router_logits, dim=-1)
    gates = F.one_hot(indices, num_classes=self._num_experts).float()
```

### 改动 4：自适应 Expert Mutation（解决 P4）

使用**相对噪声**（权重标准差的 1%）替代绝对噪声：
```python
# Expert 0 保持纯净原版能力
# Expert 1,2 注入 param.std() * 0.01 的正态噪声
with torch.no_grad():
    for i in range(1, num_experts):
        for param in self.experts[i].parameters():
            if param.is_floating_point():
                noise = torch.randn_like(param) * param.std() * 0.01
                param.add_(noise)
```

### 改动 5：修复 lb_weight 调度 floor（解决 P3 根因 2）

```python
# 原：effective_lb = lb_weight * max(0.2, 1.0 - epoch / total_epochs)
# 改：允许衰减到 0
effective_lb = lb_weight * max(0.0, 1.0 - epoch / total_epochs)
```

---

## 三阶段训练计划

### Stage 1：语义锚点（已完成，直接复用）

**无需重新训练**。直接使用 `dinov3_v2_lora/best.pt`。

此 checkpoint 包含联合优化好的：LoRA + proj1 + proj2 + head + static alpha。
这些组件在同一分布下收敛，是后续所有阶段的可信基础。

### Stage 2：MoE Bootstrap（锁死特征流形，训练路由和专家）

**核心思想**：冻结产生 z 的整条链（LoRA + proj1 + alpha + proj2），确保 experts 接收到的 z **完全等同于 v1 训练时 head 接收到的 z**。仅训练新增组件。

```bash
python src/train.py \
  --model_type dinov3_moe --num_experts 3 --lora_layers 6 --num_hooks 12 \
  --data_root /root/autodl-tmp/NTIRE_dataset \
  --output_dir /root/autodl-tmp/experiments/dinov3_moe_v4_stageB \
  --warm_start /root/autodl-tmp/experiments/dinov3_v2_lora/best.pt \
  --freeze_lora_only --freeze_proj \
  --epochs 6 --batch_size 32 --lr 3e-4 \
  --moe_lr_scale 1.0 --lora_lr_scale 0.0 \
  --lb_weight 0.02 --warmup_epochs 1 --cons_ramp_epochs 2 \
  --shards 0,1,2
```

| 组件 | 状态 | LR | 理由 |
|------|------|-----|------|
| LoRA | **frozen** | 0 | 保护 backbone 适配 |
| proj1 | **frozen** | 0 | 保护 hook 投影 |
| static alpha | **frozen** | 0 | 锁死聚合权重 |
| proj2 | **frozen** | 0 | 锁死特征流形 |
| freq_enc | train | 3e-4 | 随机初始化，从零学频域特征 |
| cls_router_proj | train | 3e-4 | 随机初始化，CLS 语义降维 |
| router | train | 3e-4 | zero-init，学习退化路由 |
| tau | train | 3e-4 | 可学习温度 |
| experts | train | 3e-4 | 从 head 克隆+自适应噪声，开始特化 |

- **不加载** `--freq_ckpt`（解决 P2：避免目标冲突）
- `lb_weight`：0.02 → 线性衰减到 0（解决 P3：允许后期特化）
- 新增 `--freeze_proj` 参数冻结 proj1 + proj2 + alpha

### Stage 3：全参数微调 + 硬路由特化

**核心思想**：所有组件已在兼容分布下各自收敛，此阶段用极低 LR 联合精调，并用 Gumbel 温度退火逼迫专家硬分流。

```bash
python src/train.py \
  --model_type dinov3_moe --num_experts 3 --lora_layers 6 --num_hooks 12 \
  --data_root /root/autodl-tmp/NTIRE_dataset \
  --output_dir /root/autodl-tmp/experiments/dinov3_moe_v4_stageC \
  --resume /root/autodl-tmp/experiments/dinov3_moe_v4_stageB/<run>/best_stage_b.pt \
  --epochs 5 --batch_size 32 --lr 8e-5 \
  --moe_lr_scale 1.0 --lora_lr_scale 0.1 \
  --lb_weight 0.005 \
  --use_gumbel --gumbel_start_epoch 2 \
  --shards 0,1,2
```

| 组件 | LR | 理由 |
|------|-----|------|
| LoRA | 8e-6 (= 8e-5 × 0.1) | 极低 LR 微调，防止破坏 backbone 适配 |
| proj1 + proj2 + alpha | 8e-5 | 低 LR 允许特征通路微调适应 MoE |
| freq_enc + cls_router_proj + router + tau | 8e-5 | 继续优化路由 |
| experts | 3e-5 | 中 LR 继续特化 |

Gumbel 温度退火调度：
- Epoch 0-1：softmax 软路由（tau 正常）
- Epoch 2：启用 Gumbel-Softmax，`gumbel_tau = 1.0`
- Epoch 3：`gumbel_tau = 0.5`
- Epoch 4：`gumbel_tau = 0.1`（接近 one-hot 硬路由）
- `lb_weight`：0.005，仅前 2 epoch 有效，之后归零

---

## 文件变更清单

| 文件 | 操作 | 变更内容 |
|------|------|----------|
| `src/models/dinov3_freq_moe.py` | **重写** | 删除 alpha_gen；保留 static alpha；新增 cls_router_proj；router 输入 128→256；Gumbel-Softmax 逻辑；自适应 mutation 噪声；推理 argmax |
| `src/train.py` | **修改** | 新增 `--freeze_proj`/`--use_gumbel`/`--gumbel_start_epoch` 参数；`cls_router_proj` 加入 `moe_new_ids`；修复 lb_weight floor→0；alpha_gen 从 moe_new_ids 移除；Gumbel tau 退火调度 |
| `src/inference.py` | 无需修改 | model.eval() 自动触发 argmax 路由 |

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| CLS.detach() 中含真伪信息 → router 走捷径（class leakage） | detach 隔离梯度；cls_router_proj 独立初始化；监控 corr(argmax_gate, label) |
| Gumbel 硬路由导致训练不稳定 | 温度退火平滑过渡 (1.0→0.1)；仅 Stage 3 后期启用 |
| Stage 2 冻结 proj 后 experts 自由度受限 | experts 有 3 层 MLP (256→256→256→1)，足够的容量做非线性特化 |
| Expert collapse | lb_weight 在 Stage 2 前期有效；自适应噪声确保初始差异 |
| Static alpha 在 Stage 3 解冻后偏移 | alpha LR 仅 8e-5，5 epoch 内偏移有限 |

---

## SESSION_ID（供 /ccg:execute 使用）
- CODEX_SESSION: 019c9961-4db6-7393-8912-eba48d80dcff
- GEMINI_SESSION: N/A (HTTP API, stateless)
