# 任务二 Track1：增量 SVD

本目录包含 Track1 的提交代码和理解版报告初稿。

## 文件说明

- `solution.cpp`：C++ 提交文件，速度优先版本。
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
10 轮 update 总耗时: 0.069 s
更新前 RMSE: 1.02171
更新后 RMSE: 1.01044
结果有效: true
```

## 提交提醒

若选择 C++ 版本提交，提交 `task2/solution.cpp`。若选择 Python 版本提交，提交 `task2/solution.py`。正式提交前建议用课程提供的本地测试脚本再跑一次，因为 Track1 的最终得分以隐藏测试集 RMSE 和计时为准。
