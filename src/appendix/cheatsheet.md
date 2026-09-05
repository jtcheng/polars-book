# 附录 B 常用操作速查表

> 按 namespace 分类编排，附常用分析操作模板（含 pandas 对照）。所有片段在 polars 1.44 实测通过。

## B.1 读取与扫描

| 操作 | Polars | pandas |
|---|---|---|
| 读 CSV | `pl.read_csv(f)` | `pd.read_csv(f)` |
| 读 CSV（显式类型） | `pl.read_csv(f, schema_overrides={"amount": pl.Float64})` | `pd.read_csv(f, dtype={"amount": float})` |
| 读 CSV（推断日期） | `pl.read_csv(f, try_parse_dates=True)` | `pd.read_csv(f, parse_dates=["ts"])` |
| 读 CSV（识别空值） | `pl.read_csv(f, null_values=["", "NULL"])` | `pd.read_csv(f, na_values=["", "NULL"])` |
| 扫描 CSV | `pl.scan_csv(f)`（返回 LazyFrame） | —（无对应） |
| 扫描 Parquet | `pl.scan_parquet(f)` | —（无对应） |
| 写 Parquet | `df.write_parquet(f)` | `df.to_parquet(f)` |
| 流式写出 | `lf.sink_parquet(f)` | —（需先物化） |
| 流式写 CSV | `lf.sink_csv(f)` | —（需先物化） |

## B.2 条件与缺失值

| 操作 | Polars | pandas |
|---|---|---|
| 条件赋值 | `pl.when(pl.col("x") > 0).then(pl.lit("high")).otherwise(pl.lit("low"))` | `np.where(df["x"] > 0, "high", "low")` |
| 填充 null | `pl.col("x").fill_null(0)` | `df["x"].fillna(0)` |
| 组内前值填充 | `pl.col("x").forward_fill().over("k")` | `df.groupby("k")["x"].ffill()` |
| 删缺失行 | `df.drop_nulls(subset=["x"])` | `df.dropna(subset=["x"])` |

## B.3 类型与转换

| 操作 | Polars | pandas |
|---|---|---|
| 宽松转换（失败变 null） | `pl.col("x").cast(pl.Int64, strict=False)` | `pd.to_numeric(df["x"], errors="coerce")` |
| 低基数转分类 | `pl.col("city").cast(pl.Categorical)` | `df["city"].astype("category")` |
| 固定类别转换 | `pl.col("dow").cast(pl.Enum(["Mon", "Tue"]))` | —（无对应） |

Categorical/Enum 选型：类别未知或会增长用 `Categorical`（跨字典 join 已自动 remap）；类别固定且已知用 `Enum`（编译期校验 + 无重编码开销）。

## B.4 字符串常用

| 操作 | Polars | pandas |
|---|---|---|
| 去首尾空白 | `pl.col("s").str.strip_chars()` | `df["s"].str.strip()` |
| 转小写 | `pl.col("s").str.to_lowercase()` | `df["s"].str.lower()` |
| 切分取元素 | `pl.col("s").str.split(".").list.last()` | `df["s"].str.split(".").str[-1]` |
| 包含判断 | `pl.col("s").str.contains("corp")` | `df["s"].str.contains("corp")` |
| 解析日期 | `pl.col("ts").str.to_datetime("%Y-%m-%d")` | `pd.to_datetime(df["ts"], format="%Y-%m-%d")` |

## B.5 聚合

| 操作 | Polars | pandas |
|---|---|---|
| 求和 | `pl.col("x").sum()` | `df["x"].sum()` |
| 均值 | `pl.col("x").mean()` | `df["x"].mean()` |
| 中位数 | `pl.col("x").median()` | `df["x"].median()` |
| 分位数 | `pl.col("x").quantile(0.9, interpolation="linear")`（polars 默认 `nearest`，跨库对齐须显式指定） | `df["x"].quantile(0.9)`（默认 linear） |
| 唯一计数 | `pl.col("x").n_unique()` | `df["x"].nunique()` |
| 近似基数（大表） | `pl.col("x").approx_n_unique()` | — |
| 直方图 | `s.hist(bin_count=10)`（Series 方法） | — |
| 连续段编号 | `pl.col("x").rle_id()` | — |

## B.6 分组

| 操作 | Polars | pandas |
|---|---|---|
| 分组聚合 | `df.group_by("k").agg(pl.col("x").sum())` | `df.groupby("k")["x"].sum()` |
| 组内环比 | `.with_columns(d=pl.col("x").pct_change().over("k"))` | `df.groupby("k")["x"].pct_change()` |
| 窗口聚合 | `pl.col("x").mean().over("k")` | `df.groupby("k")["x"].transform("mean")` |

## B.7 join

| 操作 | Polars | pandas |
|---|---|---|
| 内连接 | `df.join(other, on="k", how="inner")` | `df.merge(other, on="k")` |
| 左连接 | `how="left"` | `how="left"` |
| 全外连接 | `df.join(other, on="k", how="full", coalesce=True)`（`outer` 为旧名，已弃用） | `how="outer"` |
| 笛卡尔积 | `df.join(other, how="cross")` | `df.merge(other, how="cross")` |
| 半连接 | `how="semi"` | `df[df.k.isin(other.k)]` |
| 反连接 | `how="anti"` | `df[~df.k.isin(other.k)]` |
| 条件连接 | `df.join_where(other, pl.col("a") < pl.col("b"))` | —（无对应） |
| 时序最近邻 | `df.sort("ts").set_sorted("ts").join_asof(other, on="ts", strategy="backward")`——要求数据按 on 键有序（set_sorted 可跳过检查） | `pd.merge_asof()` |

## B.8 重塑与拼接

| 操作 | Polars | pandas |
|---|---|---|
| 垂直拼接 | `pl.concat([df1, df2])` | `pd.concat([df1, df2])` |
| 水平拼接 | `pl.concat([df1, df2], how="horizontal", strict=True)`（行数须相等） | `pd.concat([df1, df2], axis=1)` |
| 长转宽 | `long.pivot(on="metric", index="id", values="val")` | `long.pivot_table(index="id", columns="metric", values="val")` |
| 宽转长 | `wide.unpivot(index="id", on=["a", "b"])` | `df.melt(id_vars="id", value_vars=["a", "b"])` |
| 列表展开 | `df.explode("list_col")` | `df.explode("list_col")` |

## B.9 排序

| 操作 | Polars | pandas |
|---|---|---|
| 单列排序 | `df.sort("x", descending=True)` | `df.sort_values("x", ascending=False)` |
| 多列排序 | `df.sort(["k", "x"], descending=[False, True])` | `df.sort_values(["k", "x"], ascending=[True, False])` |
| 声明有序 | `df.sort("ts").set_sorted("ts")` | —（无对应） |

`set_sorted` 适用前提：数据确实已按该列有序。它是向引擎的"承诺"而非"保证"——只跳过有序性检查（省一次全表校验），实际无序会让 join_asof 等有序性敏感操作静默错配。

## B.10 时间序列

| 操作 | Polars |
|---|---|
| 环比 | `pl.col("x").pct_change(1)` |
| 同比 | `pl.col("x").pct_change(12)`（按月数据） |
| 滚动均值 | `pl.col("x").rolling_mean(7)` |
| 按时间列滚动 | `pl.col("x").rolling_mean_by("ts", "3m")` |
| 窗口对齐聚合 | `df.sort("ts").group_by_dynamic("ts", every="1h", closed="left").agg(pl.col("x").mean())` |
| 截断周期 | `pl.col("ts").dt.truncate("1h")` |
| 补齐周期 | `df.upsample("ts", every="1d")` |
| 缺失插值 | `pl.col("x").interpolate()` |
| 时间戳转日期 | `pl.from_epoch(s, time_unit="s")` |

`rolling_*_by` 与 `group_by_dynamic` 的选择：前者滑窗可重叠、逐行输出（行数不变）；后者窗口按周期边界对齐不重叠、每窗口一行（重采样成低频序列）。

## B.11 选择器

```python
import polars.selectors as cs
df.select(cs.numeric() & ~cs.first())
df.select(cs.matches(r"^amount_"))
```

## B.12 调试与调优入口

| 操作 | Polars | pandas |
|---|---|---|
| 查看查询计划 | `lf.explain()` | —（无对应） |
| 查看 schema | `lf.collect_schema()` | `df.dtypes` |
| 查看内存占用 | `df.estimated_size("mb")` | `df.memory_usage()` |

## B.13 SQL

```python
# 模块级：直接引用 Python 变量
pl.sql("SELECT city, COUNT(*) FROM df GROUP BY city").collect()

# 方法级：self 引用当前表
df.sql("SELECT * FROM self WHERE amount > 100")
```
