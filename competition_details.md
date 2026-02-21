# NTIRE 2026 比赛信息整理

## 1. 比赛概况
- **名称**: NTIRE 2026 Robust AI-Generated Image Detection in the Wild
- **平台**: [CodaBench #12761](https://www.codabench.org/competitions/12761/)
- **关联会议**: CVPR 2026 NTIRE Workshop (Denver, 2026年6月)

## 2. 时间线
| 节点 | 日期 |
|------|------|
| 训练数据发布 | 2026-01-15 |
| 测试数据发布 | 2026-03-10 |
| 提交截止 | **2026-03-17** |
| 结果公布 | 2026-03-19 |

## 3. 比赛流程
- 当前处于 **Validation Phase**：官方提供训练集 + 验证集，参赛者在验证集上提交预测结果
- 每天最多 **5 次提交**，平台自动评估并返回分数
- 无需在本地跑验证集评估，直接生成 `submission.csv` 上传即可获得反馈
- 最终 Test Phase 会发布新的测试数据

## 4. 数据集说明
- **训练集**: ~277,000 张 `.jpg` 图片，分布在 6 个 shard 中
- **图片格式**: `.jpg`，文件名为 20 位随机字符（如 `ed97447bcc3cea21bfa2.jpg`）
- **标签格式** (`labels.csv`): 含 `image_name` 和 `label` 两列
  - `0`: 真实图片
  - `1`: AI 生成图片
- **Toy Dataset**: 1,000 张无标签图片，用于快速测试推理流程
- **目录结构**:
```
data_root/
├── shard_0/
│   ├── images/*.jpg
│   └── labels.csv
├── shard_1/ ... shard_5/
└── toy_dataset/images/*.jpg
```

## 5. 评估指标
- **主指标**: **Robust ROC AUC** — 图像经过裁剪/缩放/压缩/模糊等变换后计算的 AUC
- **次指标**: **Clean ROC AUC** — 原始未变换图像上的 AUC
- 比赛排名以 Robust ROC AUC 为准

> **关键洞察**: 模型在各种图像变换后的表现比原始图像上的表现更重要。

## 6. 提交格式
- **文件名**: `submission.csv`
- **表头**: `image_name,score`
- **score**: `[0, 1]` 浮点数，越高 = 越可能是 AI 生成
- **约束**: 必须覆盖所有测试图像，不允许缺失或重复文件名
- **验证**: `python check_submission.py submission.csv`
