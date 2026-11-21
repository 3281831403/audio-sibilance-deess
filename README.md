# audio-sibilance-deess

## 题面

给人声加齿音消除后，s 音还是刺耳、整段人声却变小了：分频用一阶滤波，齿音频段泄漏 6 dB；电平按整段算，齿音定位失效；增益不按比例，超过阈值直接归零；平滑用固定常数，释放太慢出现抽吸；只压总增益，整体音量下降；齿音区间不合并，重复衰减；频段上限超奈奎斯特不报错，滤波发散；low>=high 不校验，分频无意义；批量共用滤波状态，轨道之间串；ratio<=0 不校验，除零。deess/deesser.py 是核心，tests/test_deesser.py 15个红。要修的是：按 deess/deesser.py 的契约实现 8 个函数，修掉上述 10 个缺陷，让 15 个测试全部通过。

## 运行方式

```powershell
python -m pytest tests -q
```

初始状态应恰好输出 `15 failed`，并且没有 collection、import 或 syntax error。15 条测试都会调用 `deess/deesser.py` 中的桩，当前每个桩都会抛出 `NotImplementedError`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只使用 Python 标准库（`math`、`cmath`）和 `pytest`；离线运行，不使用随机数，不 `sleep`。
- 音频一律使用 `list[float]`。
- 频段单位 Hz，电平/阈值单位 dB；重建容差 `1e-6`，无齿音直通容差 `1e-9`。
- `validate_params` 没有 `sample_rate` 参数，按 44.1 kHz 管线的奈奎斯特频率 22050 Hz 检查上限。
- `tests/test_deesser.py` 不可修改。不要通过改测试、跳过、`xfail` 或吞掉 `NotImplementedError` 来变绿；修复写在 `deess/deesser.py`。
