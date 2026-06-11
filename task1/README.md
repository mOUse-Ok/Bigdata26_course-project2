# 任务一：评分预测

本目录包含任务一的代码、实验结果和报告初稿。代码仅依赖 Python 标准库，默认读取：

- `作业说明/任务一/data/train.txt`
- `作业说明/任务一/data/test.txt`

## 目录结构

- `src/recommender.py`：数据读取、统计、验证集实验、模型训练和预测输出。
- `bin/run_task1.sh`：一键运行脚本。
- `results/predictions.txt`：对 `test.txt` 中 user-item 对的预测评分。
- `results/metrics.json`：数据统计、实验配置、验证集 RMSE/MAE、时间和内存记录。
- `results/experiment.log`：便于阅读的实验摘要。
- `report/实验报告.md`：报告初稿。

## 复现实验

在仓库根目录运行：

```bash
bash task1/bin/run_task1.sh
```

等价于：

```bash
python3 task1/src/recommender.py
```

默认配置使用正则化用户/物品偏置 SGD 模型：

- 验证集比例：20%
- 随机种子：42
- 训练轮数：35
- 学习率：0.007
- L2 正则：0.05

可选参数示例：

```bash
python3 task1/src/recommender.py --epochs 45 --lr 0.005 --reg 0.05
python3 task1/src/recommender.py --model mf --include-mf-validation
```

`--include-mf-validation` 会额外评估矩阵分解模型，运行时间明显更长。

## 提交提醒

报告中的组员姓名和学号仍是占位内容，需要提交前替换。若课程要求 Word/PDF，可将 `report/实验报告.md` 转为对应格式。
