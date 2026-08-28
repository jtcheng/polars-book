# 附录 B 常用操作速查表

> 按 namespace 分类编排，附常用分析操作模板（含 pandas 对照）。

## B.1 读取与扫描

| 操作 | Polars | pandas |
|---|---|---|
| 读 CSV | `pl.read_csv(f)` | `pd.read_csv(f)` |
| 扫描 Parquet | `pl.scan_parquet(f)` | —（无对应） |
| 流式写出 | `lf.sink_parquet(f)` | —（需先物化） |

## B.2 聚合

| 操作 | Polars | pandas |
|---|---|---|
| 求和 | `pl.col("x").sum()` | `df["x"].sum()` |
| 均值 | `pl.col("x").mean()` | `df["x"].mean()` |
| 中位数 | `pl.col("x").median()` | `df["x"].median()` |
| 分位数 | `pl.col("x").quantile(0.9)` | `df["x"].quantile(0.9)` |
| 唯一计数 | `pl.col("x").n_unique()` | `df["x"].nunique()` |
| 近似基数（大表） | `pl.col("x").approx_n_unique()` | — |
| 直方图 | `s.hist(bin_count=10)`（Series 方法） | — |
| 连续段编号 | `pl.col("x").rle_id()` | — |

## B.3 分组

| 操作 | Polars | pandas |
|---|---|---|
| 分组聚合 | `df.group_by("k").agg(pl.col("x").sum())` | `df.groupby("k")["x"].sum()` |
| 组内环比 | `.with_columns(d=pl.col("x").pct_change().over("k"))` | `df.groupby("k")["x"].pct_change()` |
| 窗口聚合 | `pl.col("x").mean().over("k")` | `df.groupby("k")["x"].transform("mean")` |

## B.4 join

| 操作 | Polars | pandas |
|---|---|---|
| 内连接 | `df.join(other, on="k", how="inner")` | `df.merge(other, on="k")` |
| 左连接 | `how="left"` | `how="left"` |
| 半连接 | `how="semi"` | `df[df.k.isin(other.k)]` |
| 反连接 | `how="anti"` | `df[~df.k.isin(other.k)]` |
| 时序最近邻 | `join_asof(strategy="nearest")` | `pd.merge_asof()` |

## B.5 时间序列

| 操作 | Polars |
|---|---|
| 环比 | `pl.col("x").pct_change(1)` |
| 同比 | `pl.col("x").pct_change(12)`（按月数据） |
| 滚动均值 | `pl.col("x").rolling_mean(7)` |
| 按时间列滚动 | `pl.col("x").rolling_mean_by("ts", "3m")` |
| 截断周期 | `pl.col("ts").dt.truncate("1h")` |
| 补齐周期 | `df.upsample("ts", every="1d")` |
| 缺失插值 | `pl.col("x").interpolate()` |
| 时间戳转日期 | `pl.from_epoch(s, time_unit="s")` |

## B.6 选择器

```python
import polars.selectors as cs
df.select(cs.numeric() & ~cs.first())
df.select(cs.matches(r"^amount_"))
```

## B.7 SQL

```python
# 模块级：直接引用 Python 变量
pl.sql("SELECT city, COUNT(*) FROM df GROUP BY city").collect()

# 方法级：self 引用当前表
df.sql("SELECT * FROM self WHERE amount > 100")
```
