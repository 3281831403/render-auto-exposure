# render-auto-exposure

## 题面

开自动曝光后，镜头一转画面就忽明忽暗、点测光还不起作用：亮度用等权平均，绿色权重错；直方图线性分桶，高光全挤在一个桶；分位数用桶下标而不是插值，误差 1 桶；测光不区分区域，点测光等于全局；适应速度不钳制，EV 跳变；EV 缩放用加而不是乘；钳制在缩放前做，先丢高光；校验不查 NaN，输出全黑；批量共用直方图，帧之间串；bins=0 不校验，除零。exposure/metering.py 是核心，tests/test_metering.py 15个红。要修的是：上述 10 个缺陷。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始实现是红态：8 个公开函数都只抛出 `NotImplementedError`，15 个测试应当全部失败，且不能出现 collection、import 或 syntax error。输出末尾应为：

```text
15 failed in ...s
```

修完之后同一命令应当变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- `tests/` 不可修改，只能修改 `exposure/metering.py` 及必要的包内辅助代码。
- 只使用 Python 标准库（`math`）和 pytest；离线运行，不使用随机数。
- 图像统一使用二维 `list[list[float]]`；线性光亮度、EV、对数直方图桶与 `1e-6` 容差是接口的一部分。
- 初始 15 个失败均来自桩函数的 `NotImplementedError`；目标不是删测试或放宽断言，而是实现契约。
