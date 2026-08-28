# 第 15 章 实战：金融时间序列回测

> 本章要解决什么问题：综合多表 join + rolling + 向量化信号计算（链式主线深度实践）。

## 15.1 场景与数据

- 输入：行情快照（分钟 K 线）、交易信号表、证券主数据
- 输出：回测收益、最大回撤、夏普比率

```python
import polars as pl

# 三张源表
bars = pl.scan_parquet("bars.parquet")        # symbol, ts, open, high, low, close, volume
signals = pl.scan_parquet("signals.parquet")  # symbol, ts, side, qty
symbols = pl.scan_parquet("symbols.parquet")  # symbol, name, sector (低基数)
```

## 15.2 多表 join 组装宽表

```python
# 交易对齐到其后最近一根 K 线（成交发生在报价之后）
bars_sorted = bars.sort("ts").set_sorted("ts")

filled = (
    signals.sort("ts").set_sorted("ts")
      .join_asof(bars_sorted, on="ts", by="symbol", strategy="backward")
      # 主数据 join：sector 是低基数 → Categorical 提速
      .join(
          symbols.with_columns(pl.col("sector").cast(pl.Categorical)).collect(),
          on="symbol", how="left",
      )
)
```

## 15.3 rolling 指标与信号计算

```python
signals = (
    pl.scan_parquet("bars.parquet")
    .with_columns(
        ma_fast=pl.col("close").rolling_mean(20).over("symbol"),
        ma_slow=pl.col("close").rolling_mean(60).over("symbol"),
        # ATR 波动率：rolling + 高低价
        atr=(((pl.col("high") - pl.col("low"))
              + (pl.col("high") - pl.col("close").shift(1)).abs()
              + (pl.col("low") - pl.col("close").shift(1)).abs()) / 3)
             .rolling_mean(14).over("symbol"),
    )
    .with_columns(
        # 布尔信号本身就是向量化表达式
        golden_cross=(pl.col("ma_fast") > pl.col("ma_slow"))
                      & (pl.col("ma_fast").shift(1) <= pl.col("ma_slow").shift(1)),
    )
)
```

## 15.4 回测统计

```python
backtest = (
    filled
    # 每笔成交盈亏：信号方向 × 价格变动
    .with_columns(
        pnl=pl.when(pl.col("side") == "BUY")
              .then(pl.col("close") - pl.col("open"))
              .otherwise(pl.col("open") - pl.col("close")) * pl.col("qty"),
    )
    # 策略日收益：分组汇总（呼应第 10 章）
    .with_columns(day=pl.col("ts").dt.truncate("1d"))
    .group_by("day")
    .agg(pl.col("pnl").sum().alias("daily_pnl"))
    .sort("day")
    .with_columns(
        ret=pl.col("daily_pnl") / pl.col("daily_pnl").shift(1).abs(),
        # 组内环比：滚动夏普
        rolling_sharpe=(pl.col("ret").mean() / pl.col("ret").std()).over(
            pl.int_range(pl.len()).floor_div(30)  # 30 日滚动窗口（示意）
        ),
    )
    # 最大回撤：累计收益的 rolling max 与当前值之差
    .with_columns(cum=pl.col("daily_pnl").cum_sum())
    .with_columns(
        drawdown=pl.col("cum") - pl.col("cum").rolling_max(252).shift(1)
    )
    .collect()
)
```

## 15.5 剖析

- rolling 的并行化收益
- 组内计算的 over 分区成本

```python
# over("symbol") 的分区数 = 证券数量（如 5000）
# 每个分区内 rolling 串行，分区间并行
# 对比：错误写法（无 over）会把所有证券的 K 线混在一个窗口里

# join_asof 的前提：双表都按 ts 排序且 set_sorted
# 忘记这一步 → 运行时报错（好）或静默错误匹配（坏）
```

## 15.6 避免逐标的循环

```python
# ❌ 反模式：5000 个标的循环回测
for symbol in symbols_list:
    df_sym = bars.filter(pl.col("symbol") == symbol)   # 5000 次全表扫描
    run_backtest(df_sym)

# ✅ 向量化：一次扫描，over 分区并行处理全部标的
bars.with_columns(
    pl.col("close").rolling_mean(20).over("symbol")
)
# 提速典型值：几十倍——不仅省了循环，还让优化器看到全貌
```

## 要点回顾

- `join_asof` 是事件对齐行情的标准工具（先排序）
- 信号 = 布尔表达式，回测 = 分组聚合——全程向量化
- `over("symbol")` 分区并行替代外层循环

## 性能检查清单

- [ ] 时间列已 `set_sorted`？（join_asof 硬性要求）
- [ ] 是否消除了逐标的 Python 循环？
- [ ] 主数据低基数列是否 cast 成 Categorical？
- [ ] rolling 窗口的行间依赖是否限制了并行度（窗口越短越好）？

## 练习

1. **金叉验证**：构造单调上升后转跌的价格序列，用 15.3 节表达式计算 golden_cross，验证信号恰好在均线交叉的那一根 K 线为 True。
2. **排序前提**：故意不执行 `set_sorted` 直接 `join_asof`，观察报错信息；修复后再验证对齐结果（每笔成交的 close 来自它之前最近一根 K 线）。
3. **回撤计算**：构造一条先涨后跌的收益曲线，用 15.4 节的 `cum_sum + rolling_max` 计算最大回撤，手工验证峰值谷值。
