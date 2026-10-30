# kairos-clean

> 一个我自己用的量化数据体检小工具。丢一个 CSV 进去，它告诉你数据哪儿有问题，
> 顺手帮你把能修的都修了。

写这个的原因很朴素：有次跑回测，收益曲线好得离谱，查了一晚上才发现是数据里有一行
`close=0`，收益率直接算成 -100% 又 +∞。后来又被负价格、重复日期、没复权的拆股坑过几次，
就想着干脆把这些检查攒成一个东西，以后拿到新数据先跑一遍。

所以它不追求"平台"或者"框架"，就三个目标：

1. **一条命令能用**，不用写配置；
2. **快**，几百万行也就几十毫秒（全 numpy，没有逐行循环）；
3. **改了什么要能看见**，每次清洗都有改动日志，不偷偷动数据。

只依赖 numpy，装起来很快（`pip install -e .` 就行）。

---

## 能查什么

| 检查项 | 说明 | 默认动作 |
|---|---|---|
| `missing_values` | 空值 / NaN | 只报告 |
| `duplicate_dates` | 同一天出现两次 | 保留最后一条 |
| `non_positive_price` | 价格 ≤ 0 | 删掉这一行 |
| `ohlc_violation` | high/low 不在 `[min(o,c), max(o,c)]` 里 | 修 high/low |
| `calendar_gap` | 相邻日期间隔超过 7 天（可能是停牌/漏行） | 只报告，可选补行 |
| `stale_price` | 连续 5 天收盘价完全一样 | 只报告 |
| `return_outlier` | 收益率的稳健 z 分数 > 8（MAD 算法） | 只报告，可选截尾 |
| `price_jump` | 单日涨跌超过 25%，并猜是不是拆股（比例接近 2/3/5/10 倍） | 只报告 |
| `zero_volume` | 成交量为 0 或负数 | 只报告 |

每个问题都会给出数量、严重度（❌ 错误 / ⚠️ 警告 / ℹ️ 提示）、几条样例，以及一句建议。

最后算一个 0~100 的质量分（按严重度加权、按行数归一），方便不同文件之间横向比。

## 怎么用

命令行：

```bash
# 只体检，不改数据
kairos-clean audit data.csv

# 体检 + 清洗，输出干净 CSV 和报告
kairos-clean clean data.csv --out-dir cleaned --fill-gaps
```

Python：

```python
import kairos_clean as kc

report = kc.audit("data.csv")
print(report["summary"]["score"])       # 98.3

result = kc.clean("data.csv", out_dir="cleaned", fill_gaps=True)
print(result.summary())                 # 改了多少行、分数涨了多少
print(result.changes[:5])               # 每条改动都能看到
```

也能直接塞一个自己拼的 Panel，不一定要读文件：

```python
from kairos_clean import Panel, audit
import numpy as np

panel = Panel("TEST", np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]"),
              {"close": np.array([10.0, 11.0])})
print(audit(panel)["summary"])
```

## 跑一下看看

仓库里有个造脏数据的脚本，直接能看效果：

```bash
python examples/demo.py
```

它会生成一份有重复日期、0 价、OHLC 矛盾、跳空、停牌、拆股的 CSV，然后体检 + 清洗，
把前后分数打出来。

真实数据上跑茅台（2608 根日线）：

```
标的 1 个 · 2,608 行 · 问题 22 个 · 质量分 98.3/100
错误 0 · 警告 22 · 提示 0
  ⚠️ calendar_gap      21  有 21 处相邻日期间隔超过 7 天
  ⚠️ return_outlier     1  有 1 个收益率的稳健 z 分数超过 8.0
```

那 21 个"缺口"基本是春节、国庆这些长假 —— 这也是它的一个短板，下面会说。

## 有多快

1,000,000 行（含 open/high/low/close/volume 五列）体检一遍：

```
1,000,000 行 体检耗时 87 ms
```

做法很简单：能向量化的地方全部向量化，连续区间用 `np.diff` + 前缀和找，不写 for 循环。

## 它会改什么、不会改什么

会改的（都可以关）：

- 重复日期：保留最后一条；
- 价格 ≤ 0 或 NaN：整行删掉；
- OHLC 矛盾：把 high 抬到 `max(o,c)`、low 压到 `min(o,c)`；
- 补缺失交易日（`--fill-gaps`，默认关）：按工作日补齐，价格前向填充，成交量补 0；
- 收益率截尾（`--winsorize`，默认关）：MAD 稳健 z 超过阈值的收益率夹到边界，同时**按比例缩放**
  open/high/low，避免把 OHLC 打乱。

**不会自动复权**。拆股只给"嫌疑"，因为复权要动整条历史序列，改错了比不改还麻烦，
这个决定应该人来做。

## 已知的问题

- **节假日会误报**：现在只按工作日判断缺行，没有接真实交易日历，所以春节会被标成
  `calendar_gap`。我也没做 A 股日历，因为维护它太麻烦了，暂时就让它报着，人眼看一眼就行。
- **只认日线**：分钟线会先被当成重复日期，得自己先聚合到日。
- **日期格式只认 ISO**：`2024-01-02` 这种，`2024/01/02` 会被跳过（会记在 `skipped_rows` 里）。
- **趋势里的正常大涨**可能被 `return_outlier` 标出来，MAD 对慢趋势不敏感，但也没那么神。
- 质量分是我拍脑袋定的权重（错误 8、警告 2、提示 0.5），不一定适合所有场景，看趋势就行。

## 代码结构

```
kairos_clean/
├── table.py    读 CSV / 写 CSV，Panel（日期 + 几列数组）
├── checks.py   九个检查，每个都是独立函数，只读不改
├── api.py      audit() / clean() / 质量分
├── report.py   Markdown / JSON 报告
└── cli.py      命令行
tests/          24 个测试，每个坑单独造一份数据
```

检查和修复是分开的：`checks.py` 只负责"发现"，`api.clean` 负责"怎么修"，
所以我后面想加检查项的时候，只要在 `checks.py` 里加一个函数就行，不用动清洗逻辑。

## 测试

```bash
python -m pytest     # 24 passed
```

每个检查项都有一份专门的脏数据；另外还测了清洗前后行数、改动日志、报告渲染和 CLI。

## License

MIT。随便用，出了事别找我（数据清洗本来就没有银弹）。