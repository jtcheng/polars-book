# 第 11 章 时间序列

> 本章要解决什么问题：掌握日期底层表示、rolling 并行化、重采样，以及同比/环比专题。

## 11.1 日期底层表示

- `Date` = i32（自纪元天数）、`Datetime` = i64（时间戳，默认微秒精度）
- 时区处理：`dt.replace_time_zone` 挂时区标签、`dt.convert_time_zone` 换算显示

这两条分别对应"存储"与"解释"两层：物理层上日期时间就是整数列，本章一切快操作的速度都源于此；语义层上时区只是挂在时间戳上的解释规则，`replace_time_zone` 与 `convert_time_zone` 的分野全在"改不改绝对时刻"。先看物理表示——用 `to_physical` 直接看存储。

```python
import polars as pl
from datetime import datetime, date

df = pl.DataFrame({
    "symbol": ["AAPL"] * 11 + ["MSFT"] * 11,   # 两个标的，各 11 个分钟点
    "ts": list(pl.datetime_range(
        datetime(2026, 8, 1), datetime(2026, 8, 1, 0, 10), "1m", eager=True
    )) * 2,
    "value": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0] * 2,
})

# 底层：Date 是天数偏移，Datetime 是时间戳整数
print(df.get_column("ts").to_physical().head(2))
# 物理表示均为整数——比较/排序/截断都是整数运算
# 实测输出（polars 1.44.1）：
# Series: 'ts' [i64]
# [1785542400000000, 1785542460000000]
#   ↑ 2026-08-01 00:00:00 / 00:01:00 的微秒时间戳（i64）

# Date 列的物理表示则是 i32 天数：
print(pl.DataFrame({"d": [date(2026, 8, 29)]}).get_column("d").to_physical())
# Series: 'd' [i32]
# [20694]   ← 自 1970-01-01 起的第 20694 天

# dt 命名空间：零成本拆解时间维度
df.with_columns(
    year=pl.col("ts").dt.year(),
    month=pl.col("ts").dt.month(),
    hour=pl.col("ts").dt.hour(),
    dow=pl.col("ts").dt.weekday(),        # 1=周一
)
```

```python
# 从整数时间戳构造日期列（日志/物联网数据常见）
pl.select(
    ts=pl.from_epoch(pl.Series([1756000000, 1756000060]), time_unit="s")
)
# Unix 秒 → Datetime，无需手动乘 1000

# 缺失时间点插值：比 forward_fill 更平滑的选项
(pl.DataFrame({"ts": [1, 2, 4], "v": [10.0, None, 40.0]})
   .with_columns(pl.col("v").interpolate())       # v: [10, 20, 40]
   # 或按另一列插值：interpolate_by("ts")
)
```

**为什么整数表示值得关心**：`ts > 某时刻` 是一次整数比较，`sort("ts")` 是整数数组排序，`dt.truncate("1h")` 是把微秒数对齐到 3 600 000 000 的整数倍——全程不碰日历库，SIMD 直接在连续整数数组上工作。本章后面会反复回到这条分界线：凡是只涉及"时刻"的操作（比较、排序、去重、join 键）都享受整数速度；凡是涉及"本地钟面"的操作（年/月/日/时拆解、时区截断）都要做日历换算，代价完全不同。

**replace_time_zone 与 convert_time_zone**：`replace_time_zone(tz)` 是"重新解释"——时钟读数不变，给这个读数挂上时区标签，绝对时刻随之改变；`convert_time_zone(tz)` 是"换算显示"——绝对时刻不变，把钟面换算到目标时区。实测对比：

```python
utc = pl.DataFrame({"ts": [datetime(2026, 8, 1, 0, 0)]}).with_columns(
    pl.col("ts").dt.replace_time_zone("UTC")
)
# 2026-08-01 00:00:00 UTC，物理整数 1785542400000000

# replace：时钟读数 00:00 不变，物理整数被改写（-8h）——"这是上海时间 00:00"
utc.with_columns(pl.col("ts").dt.replace_time_zone("Asia/Shanghai"))
# 2026-08-01 00:00:00 CST，物理整数 1785513600000000

# convert：物理整数不变（绝对时刻没动），钟面换算——"同一时刻在上海是几点"
utc.with_columns(pl.col("ts").dt.convert_time_zone("Asia/Shanghai"))
# 2026-08-01 08:00:00 CST，物理整数 1785542400000000
```

一对反差记牢即可：**replace 改时刻、convert 改读数**。给无时区数据"补"时区用 `replace_time_zone("UTC")`；把已有时区数据换到用户时区展示用 `convert_time_zone`。

**性能提示：管道内部尽量朴素 UTC，仅在展示层转换**。带时区的 `Datetime` 在逐字段拆解与截断时要考虑 DST 偏移，比朴素列贵得多——500 万行实测（polars 1.44.1，Apple M 系列，7 次取中位数）：

| 操作 | 朴素 Datetime | 带时区（Asia/Shanghai） | 差距 |
|---|---|---|---|
| `dt.truncate("1h")` | ~4.9 ms | ~303 ms | ~62× |
| `dt.hour()` | ~48.7 ms | ~110 ms | ~2.3× |
| `sort("ts")` | ~13.7 ms | ~13.7 ms | 1.0× |

排序毫无差别——它只比较物理整数，时区根本不参与；涉及"本地钟面"的操作则显著更贵（truncate 要做 DST 感知的边界换算）。实践模板：读入后统一为朴素 UTC，管道内全部按 UTC 计算，最后一步才 `convert_time_zone` 到用户时区。

## 11.2 rolling 的并行化

- `rolling_mean` / `rolling_sum` 等的窗口机制
- `rolling` vs `group_by_dynamic` 选型

窗口机制的要点是输出与输入等长：每个位置聚合自己窗口内的观测，rolling 家族永不改变行数——这一条就与"每窗一行"的重采样划清了界限。选型的第一问因此不是"用哪个函数"，而是"窗口按什么对齐"——按行数还是按时间：间隔规则时两者难分彼此，数据一有缺孔，固定行数窗口会把语义上早已"过期"的观测照常聚进来（rolling_*_by 一节有逐行对照的实例）。至于"窗口越长越慢"的直觉是否成立，下面先用一次 500 万行实测来裁决。

```python
# 滚动指标：组内并行计算
(df.sort("ts")
   .with_columns(
       ma3=pl.col("value").rolling_mean(3).over("symbol"),
       std3=pl.col("value").rolling_std(3).over("symbol"),
   ))
# 窗口越短，行间依赖越弱，并行度越高

# 时间窗口（而非固定行数）：rolling_*_by 系列按另一列的值定义窗口
df.with_columns(
    pl.col("value").rolling_sum_by("ts", window_size="3m").over("symbol").alias("sum_3m")
)
```

### 窗口长度 vs 并行度：一次 500 万行实测

直觉：窗口长度为 w 时，第 i 行的输出依赖前 w 行——窗口越长，行间依赖链越长，能安全并行切分的位置就越少，理应越慢。第 7 章练习 1 正是从"行间依赖"的角度让你验证这一点。实测（polars 1.44.1，Apple M 系列，5 次取中位数）：

```python
import time
import statistics
import polars as pl

N = 5_000_000
s = pl.DataFrame({"v": pl.int_range(0, N, eager=True, dtype=pl.Int64) * 3})

def bench(window):
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        s.select(pl.col("v").rolling_mean(window))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

for w in [3, 100, 1000, 10_000]:
    print(f"rolling_mean({w:>5}): {bench(w) * 1000:5.1f} ms")
# 实测输出：
# rolling_mean(    3):  51.8 ms
# rolling_mean(  100):  52.0 ms
# rolling_mean( 1000):  51.4 ms
# rolling_mean(10000):  52.2 ms
# （复测波动 ±4%，窗口之间的差异始终小于噪声）
```

窗口从 3 拉到 10000（三千多倍），耗时纹丝不动。原因是引擎优化：`rolling_mean(w)` 并不逐窗口重算，而是**增量维护滑动和**——窗口每右移一行，累加器"进一个元素、出一个元素"，每个输出行摊销 O(1)，窗口长度根本不进入时间复杂度。反证很直接：若按朴素的 O(n·w) 逐窗口求和，500 万行 × 10000 窗口 = 500 亿次元素操作，52 ms 内无论如何做不完——实测耗时本身就是复杂度为 O(n) 的证据。并行切分同理：线程按行块划分、每个块独立维护自己的增量状态，切分点数量与窗口长度无关。

诚实结论：**"窗口越长越慢"的直觉在当前引擎（1.44.1）下不成立**，不必为缩短窗口做无谓的优化。直觉负责提出假设，实测负责裁决——这正是第 12 章方法论的核心。

### rolling / rolling_*_by / group_by_dynamic 选型

| 需求 | 工具 | 语义 | 对齐方式 | 典型场景 |
|---|---|---|---|---|
| 固定行数窗口 | `rolling_*` | "最近 N 行" | 按行号，每个输出恰好覆盖 N 个观测 | 均匀采样序列的技术指标（MA5、MA20） |
| 时间边界窗口（逐行输出） | `rolling_*_by` | "过去 T 时间内的所有观测" | 按时间列的值，窗口内行数不固定 | 不规则间隔日志的"过去 5 分钟错误率" |
| 变频聚合（每窗一行） | `group_by_dynamic` | "每个 T 周期聚合出一个值" | 按周期边界对齐，窗口互不重叠 | 分钟 → 小时 OHLC 重采样 |

分水岭一句话：**rolling 数行、rolling_*_by 数时间、group_by_dynamic 切周期**。数据间隔均匀且无缺孔时，前两者的结果几乎一致；间隔一旦不规则，就必须想清楚要的是哪种语义。

### rolling_*_by：按时间列定义窗口

`rolling_sum_by("ts", "3m")` 与 `rolling_sum(3)` 的差异不止"按时间"三个字：固定行数窗口下，每个输出恰好聚合 3 个观测，与它们发生在多久之前无关，且开头 2 行因窗口不满输出 null；时间窗口下，聚合的是"过去 3 分钟内"的观测——窗口里有多少行随间隔浮动，窗口不满时默认仍输出（`min_periods=1`）。不规则间隔下两者的分野一目了然：

```python
from datetime import datetime

irr = pl.DataFrame({
    "ts": [
        datetime(2026, 8, 1, 0, 0),
        datetime(2026, 8, 1, 0, 1),    # 间隔 1 分钟
        datetime(2026, 8, 1, 0, 10),   # 间隔 9 分钟——跨过固定行数窗口，也滑出时间窗口
        datetime(2026, 8, 1, 0, 11),
    ],
    "v": [10.0, 20.0, 30.0, 40.0],
})
irr.with_columns(
    rows3=pl.col("v").rolling_sum(3),                            # [null, null, 60, 90]
    time3m=pl.col("v").rolling_sum_by("ts", window_size="3m"),   # [10, 30, 30, 70]
)
# 00:10 那一行：rows3 = 10+20+30 = 60（最近 3 行——隔着 9 分钟也照算不误）
#              time3m = 30（3 分钟窗内只剩自己——20 在 9 分钟前已被滑出）
```

## 11.3 重采样

```mermaid
flowchart LR
    A["原始高频数据<br/>（每分钟）"] --> B["upsample / truncate<br/>对齐到周期边界"]
    B --> C["group_by_dynamic<br/>1h 聚合"]
    C --> D["输出低频序列<br/>（每小时 OHLC）"]
```

```python
# group_by_dynamic：窗口对齐聚合——分钟 → 小时 OHLC（group_by 按标的分区）
(df.sort("ts")
   .group_by_dynamic("ts", every="1h", closed="left", group_by="symbol")
   .agg(
       open=pl.col("value").first(),
       high=pl.col("value").max(),
       low=pl.col("value").min(),
       close=pl.col("value").last(),
   ))

# upsample：补齐缺失时间点（行数可能膨胀）
df.upsample("ts", every="1m", group_by="symbol").with_columns(
    pl.col("value").forward_fill().over("symbol"),   # 组内前值填充
)
```

## 11.4 同比/环比专题

```python
# 环比：按月分组总额
monthly = (
    df.group_by(pl.col("ts").dt.truncate("1mo").alias("month"))
      .agg(pl.col("amount").sum().alias("total"))
      .sort("month")
)
monthly.with_columns(
    prev_total=pl.col("total").shift(1),
    mom=pl.col("total").pct_change(1),          # 环比（Month-over-Month）
    yoy=pl.col("total").pct_change(12),         # 同比（Year-over-Year）
)
```

- `shift` + 分组：`over` 窗口下的组内环比
- `pct_change(n)`：相对 n 期前的变化率
- 分组内 period-over-period（按年分组的同比）
- 缺失周期补齐：`upsample` 后再算环比，避免错位比较

环比的算术本体就是 shift：把上一期的值平移到当前行，两行相除。`pct_change(n)` 是它的封装——相对 n 期前的变化率，n=12 就是月度数据的同比。分组场景的全部要点在 `over`：shift 必须被限制在组内，否则每组第一期的"上一期"会错拿相邻用户的数据。而最隐蔽的坑是周期缺失：`pct_change` 只看行位置、不看日历，缺了 2 月，1 月的下一行就是 3 月——一个"看似正常"的 30% 其实是两个月累计涨幅，所以先 `upsample` 补齐周期再算环比是正确性前提。

```python
# 组内环比：每个用户自己的月度趋势
(df.with_columns(
    month=pl.col("ts").dt.truncate("1mo"),
  )
  .group_by("user", "month")
  .agg(pl.col("amount").sum().alias("total"))
  .sort("user", "month")
  .with_columns(
      # shift + over：上一期取的是"同一用户"的上一期
      prev=pl.col("total").shift(1).over("user"),
      mom=pl.col("total").pct_change(1).over("user"),
  ))
```

```python
# 陷阱：缺失周期会让环比错位——1 月和 3 月比，跳过了 2 月
sparse = pl.DataFrame({
    "month": ["2026-01", "2026-03"],   # 2 月数据缺失
    "total": [100, 130],
})
sparse.with_columns(pl.col("total").pct_change(1))
# 得 0.3——一个"看似正常"的 30% 环比，实际是跨了两个月的错位比较：
# 它把 1→3 月的总涨幅记在了一个月头上，掩盖了真实走势

# 假如 2 月的真实值是 115（从源表找回），补齐后环比完全不同
dense = pl.DataFrame({
    "month": ["2026-01", "2026-02", "2026-03"],
    "total": [100, 115, 130],
})
dense.with_columns(pl.col("total").pct_change(1))
# 2 月 0.15、3 月 0.13——涨势平缓得多，0.3 是缺月制造的错觉

# 对策：周期补齐后再计算（拿不到真实值时用 forward_fill 近似）
filled = (sparse.with_columns(pl.col("month").str.to_date("%Y-%m"))
                .upsample("month", every="1mo")
                .with_columns(pl.col("total").forward_fill()))
filled.with_columns(pl.col("total").pct_change(1))
```

## 要点回顾

- 环比本质是 shift；分组内环比本质是 over + shift
- 时间序列操作前先 `set_sorted`
- 缺失周期会导致 pct_change 错位——先 upsample 补齐

## 性能检查清单

- [ ] 时间列是否已排序并 `set_sorted`？
- [ ] 环比计算是否补齐了缺失周期？
- [ ] rolling 窗口是否用 `over` 正确限定了分组？
- [ ] 高基数分组（如 per-user rolling）是否考虑了分区代价？

## 练习

1. **OHLC 重采样**：生成一周的分钟级随机价格，用 `group_by_dynamic` 聚合成小时级 OHLC，验证每个窗口的 open 是第一个值、close 是最后一个值。
2. **错位陷阱复现**：构造缺 2 月的月度数据，先直接 `pct_change(1)` 得到错误环比，再用 `upsample + forward_fill` 修正，对比两个结果。
3. **组内环比**：对多用户月度数据计算每个用户的环比（`over` + `pct_change`），验证用户边界处第一期的环比为 null 而不是错拿别的用户的值。
