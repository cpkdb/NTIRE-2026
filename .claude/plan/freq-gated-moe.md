# Frequency-Gated MoE 实施计划

## 架构: DINOv3FreqMoE

将 DINOv3 的单一分类头替换为 **Mixture-of-Experts** 头，由 Haar DWT 高频子带提取的 **频率 Token** 进行门控路由。频率编码器被吸收进 DINOv3 模型内部，消除独立的 FreqClassifier 分支。

---

## 1. 新类: `DINOv3FreqMoE` (继承 `DINOv3Classifier`)

### 1.1 频率编码器 (复用 FreqClassifier 内部结构)
- 输入: 归一化图像 `x`（模型内部反归一化回原始像素空间）
- Haar DWT → 9通道高频子带 (每个RGB通道的 LH/HL/HH)
- CNN 编码器: `Conv2d(9→32→64→96→128)` + BN+ReLU, stride-2 下采样
- `AdaptiveAvgPool2d(1)` → flatten → **128维频率 Token** `f_tok`

```python
def _freq_token(self, x):
    # x: [B,3,H,W] 归一化后的输入
    raw = x * self._std + self._mean  # 反归一化
    c = F.conv2d(raw, self._haar.to(raw.dtype), stride=2, groups=3)  # [B,12,H/2,W/2]
    hf = torch.cat([c[:, 1:4], c[:, 5:8], c[:, 9:12]], dim=1)  # [B,9,H/2,W/2]
    return self.freq_pool(self.freq_enc(hf)).flatten(1)  # [B,128]
```

### 1.2 路由器
- 输入: 频率 Token `f_tok` [B,128]
- `Linear(128,64)` → `ReLU` → `Linear(64, num_experts)` → `Softmax(dim=-1, /tau)` → 门控权重 [B, E]
- **Zero-init** 路由器输出层（初始均匀路由）
- `tau` = 可学习参数, 初始值=1.0, 限制范围 [0.1, 10.0]

```python
self.router = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_experts))
nn.init.zeros_(self.router[-1].weight)
nn.init.zeros_(self.router[-1].bias)
self.tau = nn.Parameter(torch.ones(1))
```

### 1.3 专家池
- `num_experts = 3` (可配置, 默认3)
  - 依据: Codex 建议4, Gemini 建议3 (通用型 / 高频GAN专家 / 低频Diffusion专家)。135K 数据集下3个专家可降低坍缩风险。
- 每个专家: `MLP(256→256→256→1)` (与现有 `self.head` 结构一致)
- **热启动**: 从预训练 `self.head` 权重克隆所有专家
- 密集软路由: `logit = Σ(gate_i * expert_i(z))`，其中 `z` 为 DINOv3 特征

```python
self.experts = nn.ModuleList([
    copy.deepcopy(self.head) for _ in range(num_experts)
])
```

### 1.4 动态 Alpha (频率感知的 RINE 聚合)
用频率条件化的动态权重替换静态 `self.alpha`:
- `f_tok [B,128]` → `Linear(128, n_hooks * proj_dim)` → reshape → `[B, n_hooks, proj_dim]`
- 替代静态 alpha 进行 softmax 加权 hook 聚合
- 效果: 噪声/压缩图像自动提升深层 hook 权重；干净图像依赖浅层 hook 进行精细伪影检测

```python
self.alpha_gen = nn.Linear(128, n_hooks * proj_dim)
# forward 中:
alpha_dyn = self.alpha_gen(f_tok).view(B, n_hooks, proj_dim)
z = (torch.softmax(alpha_dyn, dim=1) * g).sum(dim=1)
```

### 1.5 前向流
```
x → [反归一化 + Haar DWT + CNN enc + pool]         → f_tok [B,128]  (频率 Token, 先计算)
x → [backbone + hooks + proj1]                      → g [B,H,256]    (hook 特征)
f_tok → [alpha_gen] → softmax → 动态 alpha [B,H,256]               (频率感知聚合)
(alpha * g).sum(1) → proj2                          → z [B,256]      (聚合特征)
f_tok → [router / tau] → softmax                    → gates [B,3]    (专家路由)
z → [expert_0(z), ..., expert_2(z)] → stack          → expert_logits [B,3]
logit = (gates * expert_logits).sum(dim=-1)          [B]
```

### 1.6 辅助输出
`forward_with_aux(x)` 返回 `(logit, z, gates, expert_logits)`:
- `z` 用于对比损失
- `gates` 用于负载均衡损失
- 专家诊断日志

---

## 2. 负载均衡损失

防止专家坍缩:
```python
def load_balance_loss(gates):
    # gates: [B, E]
    avg_gate = gates.mean(dim=0)  # [E]
    return gates.shape[1] * (avg_gate ** 2).sum()
```

整合到训练中: `loss = loss_cls + λ_cont * loss_cont + λ_lb * loss_lb`
- `λ_lb = 0.01` (可调, Gemini 建议 0.05; 保守起步)

---

## 3. 训练策略 (两阶段)

### Stage A: MoE 头部训练 (6 epochs)
- **冻结**: backbone (始终), LoRA 权重, proj1, proj2, alpha
- **训练**: freq_enc, freq_pool, router, tau, experts
- LR: 3e-4, batch_size=32
- 目的: 让路由器和专家在不干扰预训练特征的情况下完成专业化

### Stage B: 端到端微调 (10 epochs)
- **解冻**: LoRA 权重 (通过参数组设置 10x 更低的 LR)
- **训练**: freq_enc, freq_pool, router, tau, experts, LoRA
- LR: 3e-5 (LoRA: 3e-6)
- 目的: 频率路由与空间特征协同适配

### 检查点: 从 `/root/autodl-tmp/experiments/dinov3_v2_lora/best.pt` 热启动

---

## 4. 文件变更

### 4.1 新建: `src/models/dinov3_freq_moe.py`
- 类 `DINOv3FreqMoE(DINOv3Classifier)`
- 方法: `__init__`, `_freq_token`, `forward`, `forward_with_aux`, `get_features`, `trainable_params`, `save_trainable`
- 新增组件: `freq_enc`, `freq_pool`, `_haar`/`_mean`/`_std` 缓冲区, `alpha_gen`, `router`, `tau`, `experts`

### 4.2 修改: `src/models/__init__.py`
- 添加导入: `from .dinov3_freq_moe import DINOv3FreqMoE`
- 添加到 `__all__`

### 4.3 修改: `src/train.py`
- `build_model()`: 添加 `elif args.model_type == "dinov3_moe"` 分支
- `model_type` choices: 添加 `"dinov3_moe"`
- `train_one_epoch()`: 当模型有 `forward_with_aux` 时添加负载均衡损失
- 添加 `--lb_weight` 参数 (默认 0.01)
- 添加 `--freeze_backbone_features` 标志 (Stage A 用)
- 添加 `--lora_lr_scale` 参数 (Stage B 参数组用)

### 4.4 修改: `src/inference.py`
- `load_model()`: 添加 `elif model_type == "dinov3_moe"` 分支
- `model_type` choices: 添加 `"dinov3_moe"`

---

## 5. 训练命令

### Stage A
```bash
python src/train.py \
  --model_type dinov3_moe \
  --data_root /root/autodl-tmp/NTIRE_dataset/train \
  --output_dir /root/autodl-tmp/experiments/dinov3_moe_v1 \
  --epochs 6 --batch_size 32 --lr 3e-4 \
  --lora_layers 6 --lb_weight 0.01 \
  --freeze_backbone_features \
  --resume_from /root/autodl-tmp/experiments/dinov3_v2_lora/best.pt \
  --consistency_weight 0.5 --label_smoothing 0.05
```

### Stage B
```bash
python src/train.py \
  --model_type dinov3_moe \
  --data_root /root/autodl-tmp/NTIRE_dataset/train \
  --output_dir /root/autodl-tmp/experiments/dinov3_moe_v1 \
  --epochs 10 --batch_size 32 --lr 3e-5 \
  --lora_layers 6 --lora_lr_scale 0.1 --lb_weight 0.01 \
  --resume /root/autodl-tmp/experiments/dinov3_moe_v1/last.pt \
  --consistency_weight 0.5 --label_smoothing 0.05
```

---

## 6. 风险缓解

| 风险 | 缓解措施 |
|------|---------|
| 专家坍缩 (所有流量涌入1个专家) | 负载均衡损失 + 路由器 zero-init + 温度缩放 |
| 频率编码器过拟合 | Stage A 冻结策略 + BN momentum |
| 预训练 DINOv3 性能回退 | 从现有 head 热启动专家; Stage A 隔离影响 |
| 训练不稳定 | 梯度裁剪 (max_norm=1.0), 余弦退火 |
| 显存开销 | Freq CNN ~0.2M 参数; 3 专家共享维度 → 相比 backbone 可忽略 |

---

## 7. 参数预算

| 组件 | 参数量 |
|------|--------|
| 频率编码器 (CNN 9→128) | ~175K |
| 路由器 (128→64→3) | ~8.4K |
| Alpha 生成器 (128→n_hooks*256) | ~394K |
| 3 个专家 (256→256→256→1 每个) | ~396K (3 × 132K) |
| **新增参数总计** | **~974K** |
| DINOv3 LoRA (已有) | ~200K |
| **可训练参数总计** | **~1.9M** (vs 304M backbone 冻结) |

---

## 8. 双模型交叉验证总结

| 决策点 | Codex | Gemini | 最终决策 |
|--------|-------|--------|---------|
| 专家数量 | 4 | 3 | **3** (135K 数据集下降低坍缩风险) |
| 频率编码器训练 | 端到端+热启动 | 端到端+热启动 | **端到端+热启动** (共识) |
| Alpha 聚合 | 静态 (保持现有) | 动态 (频率条件化) | **动态** (严格更优, 频率感知) |
| LB 权重 | 0.01 | 0.05 | **0.01** (保守起步, 熵下降时上调) |
| 交叉注意力 vs 路由 | 仅路由 | 路由 + 动态 Alpha | **两者结合** |
| 训练阶段 | 两阶段 (6+10) | 两阶段 | **两阶段** (共识) |
| 是否值得冒险? | 是 | 是 | **是** (共识) |

---

## 执行
```
/ccg:execute .claude/plan/freq-gated-moe.md
```
