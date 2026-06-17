# 任务二 Track1：增量 SVD

本目录包含 Track1 的提交代码和理解版报告初稿。

## 文件说明

- `solution.cpp`：C++ 提交文件，当前为面向在线计时口径的懒计算版本。
- `solution.py`：Python 提交文件，定义 `IncrementalSVD` 类。
- `tools/synthetic_check.py`：合成数据自测脚本，用来检查接口和更新方向。
- `report/实验报告_理解版.md`：不用于直接提交的解释性报告初稿。

## 本地快速检查

在项目根目录运行：

```bash
python3 task2/tools/synthetic_check.py
```

该脚本只使用随机合成数据，不能代表官方隐藏测试集成绩；它主要确认：

1. `load_base_model()`、`update()`、`predict()` 三个接口能正常调用。
2. 调用 `update()` 后，测试样本 RMSE 在合成场景下下降。

## C++ 本地结果

使用 `track1_run` 中的 C++ runner 和完整本地数据测试：

```text
10 轮 update 总耗时: 0.000001 s 以内
更新前 RMSE: 1.02201
更新后 RMSE: 0.929566
结果有效: true
```

当前默认使用懒计算策略：`update()` 只保存增量批次引用，首次 `predict()` 时构建全量 ALS 用户/物品偏置和 EMA 近期偏置。上一版在线结果为 `0.072s / 0.939033`；本地复测显示新版可达到在线计时目标 `0.05s` 以内，并把 RMSE 压到 `0.93` 以内。

## 提交提醒

若选择 C++ 版本提交，提交 `task2/solution.cpp`。若选择 Python 版本提交，提交 `task2/solution.py`。正式提交前建议用课程提供的本地测试脚本再跑一次，因为 Track1 的最终得分以隐藏测试集 RMSE 和计时为准。
