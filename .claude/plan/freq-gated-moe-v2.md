# DINOv3FreqMoE v3.0: Warm Start + Expert Mutation

## 版本历史
- **v1.0**: Warm start (从 DINOv3+LoRA 克隆 head→experts)，Codabench 0.90+
- **v2.0**: 冷启动独立训练，shard 0,1 / shard 2 验证，Codabench 0.86
- **v2.1**: 冷启动 + 全量训练 + RandomResizedCrop + 降低增强，Codabench 待定
- **v3.0**: 回归 warm start + expert mutation + 频率安全增强 + 分组 LR

---

## v2.1 失败根因 (Codabench 0.86 vs v1 的 0.90+)

### 1. MoE 鸡与蛋陷阱
随机初始化的 3 个专家输出垃圾梯度，router 无法学到有意义的路由。
Warm start 时专家已具备强分类能力，router 学习难度指数级下降。

### 2. RandomResizedCrop 破坏频域取证信号
Haar DWT 高频子带对空间频率有固定响应带宽。随机 crop/scale 导致同一伪影在频域中位置不确定，freq_enc 无法学到稳定 pattern。

### 3. LoRA 知识遗忘
v1 的 LoRA 经过 14 epoch AIGI 适配，已将 DINOv3 语义空间扭转为伪影敏感空间。
冷启动从 zero-init lora_B 开始，4+10 epoch 不足以恢复同等敏感度。

### 4. alpha_gen 冷启动
v1 用静态 alpha（已学好），v2 的 alpha_gen 随机初始化，与随机专家耦合导致优化更困难。

---

## 架构 (不变)

DINOv3 backbone (frozen) + LoRA + Haar DWT freq_enc → 128-dim f_tok → alpha_gen 动态聚合 12 hook → 3-expert MoE head (freq-gated soft routing)

---

## v3.0 变更

### 变更 1: 恢复 Warm Start
- 移除 `--warm_start/--freq_ckpt` 禁令
- 加载 `dinov3_v2_lora/best.pt`（14 epoch AIGI 适配的 LoRA + proj + head）
- 加载 `freq_v2/best.pt`（预训练频率编码器）
- `load_trainable()` 自动将 head.* 克隆到 3 个 experts

### 变更 2: Expert Mutation
- Expert 0 保持原始 head 权重（锚点）
- Expert 1, 2 注入 1e-3 高斯噪声打破对称性
- 配合 router zero-init，初始均匀路由但专家有微小差异

### 变更 3: 回退 Resize(224,224)
- RobustTransform light/strong 和 non-robust 路径全部回退到 `T.Resize((224,224))`
- 保留 `RandomDownsampleUpsample` 作为频率安全的多样性来源
- 保留 v2.1 的降低增强参数（JPEG q30+, noise 0.04 等）

### 变更 4: 分组 LR + 独立冻结控制
新增参数：
- `--freeze_experts`: 独立控制专家冻结
- `--moe_lr_scale`: router/alpha_gen/freq_enc 的 LR 倍率

3 组参数分组：
| 组 | 包含 | Stage A LR | Stage B LR |
|----|------|-----------|-----------|
| moe_new | router, alpha_gen, freq_enc, tau | lr * moe_lr_scale | lr * moe_lr_scale |
| base | proj1, proj2, experts | lr | lr |
| lora | LoRA A/B | frozen | lr * lora_lr_scale |

---

## 训练命令

### Stage A (2 epochs)
```bash
python src/train.py \
  --model_type dinov3_moe \
  --data_root /root/autodl-tmp/NTIRE_dataset \
  --output_dir /root/autodl-tmp/experiments/dinov3_moe_v3 \
  --epochs 2 --batch_size 32 --lr 3e-4 \
  --lora_layers 6 --num_hooks 12 \
  --lb_weight 0.03 --warmup_epochs 1 --cons_ramp_epochs 2 \
  --freeze_lora_only \
  --shards 0,1,2 \
  --warm_start /root/autodl-tmp/experiments/dinov3_v2_lora/best.pt \
  --freq_ckpt /root/autodl-tmp/experiments/freq_v2/best.pt \
  --consistency_weight 0.1 --label_smoothing 0.05
```

### Stage B (3 epochs)
```bash
python src/train.py \
  --model_type dinov3_moe \
  --data_root /root/autodl-tmp/NTIRE_dataset \
  --output_dir /root/autodl-tmp/experiments/dinov3_moe_v3_stageB \
  --epochs 3 --batch_size 32 --lr 5e-5 \
  --lora_layers 6 --num_hooks 12 \
  --lora_lr_scale 0.1 --moe_lr_scale 2.0 \
  --lb_weight 0.01 --warmup_epochs 0 --cons_ramp_epochs 1 \
  --shards 0,1,2 \
  --consistency_weight 0.5 --label_smoothing 0.05 \
  --resume <Stage_A_best.pt>
```

---

## 文件变更清单 (v3.0)

### `src/models/dinov3_freq_moe.py`
- `load_trainable()`: head→experts 克隆时 expert 1,2 注入 1e-3 噪声

### `src/train.py`
- 移除 warm_start/freq_ckpt 禁令
- build_model 后加回 warm_start + freq_ckpt 加载
- 新增 `--freeze_experts`, `--moe_lr_scale` 参数
- 3 组分组 LR (moe_new / base / lora)

### `src/transforms.py`
- RobustTransform light/strong: `RandomResizedCrop` → `Resize((224,224))`
- `get_train_transform` non-robust: `RandomResizedCrop` → `Resize((224,224))`
- 保留 v2.1 的降低增强参数不变
