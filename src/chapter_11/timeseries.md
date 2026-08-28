# 第 11 章 时间序列

> 本章要解决什么问题：掌握日期底层表示、rolling 并行化、重采样，以及同比/环比专题。

## 11.1 日期底层表示

- `Date` = i32（自纪元天数）、`Datetime` = i64（时间戳）
- 时区处理：`dt.convert_time_zone`

```python
import polars as pl
from datetime import datetime

df = pl.DataFrame({
    "ts": pl.datetime_range(
        datetime(2026, 8, 1), datetime(2026, 8, 1, 0, 10), "1m", eager=True
    ),
    "value": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
})

# 底层：Date 是天数偏移，Datetime 是时间戳整数
print(df.get_column("ts").to_physical().head(2))
# 物理表示均为整数——比较/排序/截断都是整数运算

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

## 11.2 rolling 的并行化

- `rolling_mean` / `rolling_sum` 等的窗口机制
- `rolling` vs `group_by_dynamic` 选型

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
    pl.col("value").rolling_sum_by("ts", window_size="3m").alias("sum_3m")
)
```

## 11.3 重采样

```mermaid
flowchart LR
    A["原始高频数据<br/>（每分钟）"] --> B["upsample / truncate<br/>对齐到周期边界"]
    B --> C["group_by_dynamic<br/>1h 聚合"]
    C --> D["输出低频序列<br/>（每小时 OHLC）"]
```

```python
# group_by_dynamic：窗口对齐聚合——分钟 → 小时 OHLC
(df.sort("ts")
   .group_by_dynamic("ts", every="1h", closed="left")
   .agg(
       open=pl.col("value").first(),
       high=pl.col("value").max(),
       low=pl.col("value").min(),
       close=pl.col("value").last(),
   ))

# upsample：补齐缺失时间点（行数可能膨胀）
df.upsample("ts", every="1m").with_columns(
    pl.col("value").forward_fill()   # 通常紧跟填充
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
    "month": ["2026-01", "2026-03"],   # 缺 2 月
    "total": [100, 130],
})
sparse.with_columns(pl.col("total").pct_change(1))  # 30%——但它不是环比！

# 对策：先补齐再计算
dense = sparse.with_columns(pl.col("month").str.to_date("%Y-%m"))
dense = dense.upsample("month", every="1mo").with_columns(
    pl.col("total").forward_fill()
)
dense.with_columns(pl.col("total").pct_change(1))  # 正确的逐月环比
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
