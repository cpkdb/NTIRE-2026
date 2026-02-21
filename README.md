# NTIRE 2026 Robust AI-Generated Image Detection in the Wild

比赛详情见 [competition_details.md](competition_details.md) | Baseline 调研见 [docs/baseline_survey.md](docs/baseline_survey.md)

## 项目结构

```
/root/NTIRE/
├── baselines/
│   ├── FatFormer/                  # CVPR 2024 - 频率域+文本对齐
│   └── rine/                       # ECCV 2024 - CLIP中间层聚合
├── src/
│   ├── config.py                   # 配置管理
│   ├── transforms.py               # 鲁棒性数据增强 (JPEG/模糊/噪声/缩放)
│   ├── datasets/
│   │   └── aigi_dataset.py         # 数据集加载 (自动适配 shard 目录结构)
│   ├── models/
│   │   ├── base_classifier.py      # 模型基类 + timm 封装
│   │   └── rine_wrapper.py         # RINE 模型封装
│   ├── train.py                    # 训练入口 (支持 RINE/timm)
│   ├── inference.py                # 推理 (支持集成/TTA)
│   └── utils/
│       └── metrics.py              # ROC AUC 计算
├── scripts/
│   └── prepare_submission.sh       # 提交准备脚本
├── check_submission.py             # 官方提交格式验证
├── competition_details.md          # 比赛细则
├── docs/
│   └── baseline_survey.md          # Baseline 调研报告
├── experiments/                    # 实验输出
├── submissions/                    # 提交文件
└── requirements.txt
```

## 解决方案

### 核心思路

主指标为 **Robust ROC AUC**（图像经变换后的检测能力），因此方案围绕鲁棒性设计：

1. **主模型 RINE**: 冻结 CLIP ViT-L/14，聚合所有中间层 CLS token，仅训练轻量 head
2. **鲁棒增强**: 50% 轻增强（保留伪造痕迹）+ 50% 强增强（JPEG压缩/模糊/噪声/缩放）
3. **对比学习**: BCE + 对比损失，增强特征空间的类间分离
4. **推理增强**: TTA（原图+翻转+模糊）+ 多模型集成

### 训练

```bash
cd /root/NTIRE/src

# RINE 模型 (默认鲁棒增强)
python train.py \
    --data_root /root/autodl-tmp/NTIRE_dataset \
    --model_type rine --backbone "ViT-L/14" \
    --epochs 10 --batch_size 16 --lr 1e-4

# timm 模型 (如 ConvNeXt，用于集成)
python train.py \
    --data_root /root/autodl-tmp/NTIRE_dataset \
    --model_type timm --model convnext_base \
    --epochs 10 --batch_size 32
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | `rine` | 模型类型: `rine` / `timm` |
| `--backbone` | `ViT-L/14` | RINE 的 CLIP backbone |
| `--model` | `resnet50` | timm 模型名称 |
| `--no_robust_aug` | `False` | 禁用鲁棒增强 |
| `--epochs` | `10` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `1e-4` | 学习率 |
| `--val_ratio` | `0.1` | 验证集比例 |
| `--output_dir` | `/workspace/experiments` | 输出目录 |
| `--resume` | `None` | 断点续训 |

### 推理与提交

```bash
# 单模型推理
python inference.py \
    --data_root /path/to/test_data \
    --checkpoint /workspace/experiments/best.pt \
    --model_type rine --output submission.csv

# TTA 推理 (原图 + 水平翻转 + 轻微模糊 取平均)
python inference.py \
    --data_root /path/to/test_data \
    --checkpoint best.pt --model_type rine --tta

# 多模型集成
python inference.py \
    --data_root /path/to/test_data \
    --ensemble --model_paths "rine_best.pt,convnext_best.pt" \
    --model_type rine
```

### 验证提交格式

```bash
python check_submission.py submission.csv
```

## 比赛提交流程

1. 训练模型 -> 生成 `submission.csv`
2. `python check_submission.py submission.csv` 验证格式
3. 上传到 [CodaBench](https://www.codabench.org/competitions/12761/) (每天最多 5 次)
4. 平台自动评估，返回 Robust AUC / Clean AUC 分数

## 依赖安装

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git  # RINE 需要
```
