# 第 15 章 实战：金融时间序列回测

> 本章要解决什么问题：综合多表 join + rolling + 向量化信号计算（链式主线深度实践）。

## 15.1 场景与数据

- 输入：行情快照（日线 K 线）、交易信号表、证券主数据
- 输出：回测收益、最大回撤、夏普比率

```python
import polars as pl

# 三张源表
bars = pl.scan_parquet("bars.parquet")        # symbol, ts, open, high, low, close, volume
signals = pl.scan_parquet("signals.parquet")  # symbol, ts, side, qty
symbols = pl.scan_parquet("symbols.parquet")  # symbol, name, sector (低基数)
```

先运行一次下面的数据生成（2 个 symbol × 500 根日线 K 线），本章代码即可顺序执行：

```python
from datetime import datetime
import numpy as np

rng = np.random.default_rng(42)
ts = pl.datetime_range(
    datetime(2024, 8, 1), datetime(2026, 12, 31), "1d", eager=True
).head(500)

frames = []
for sym in ["AAPL", "MSFT"]:
    n = len(ts)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))   # 随机游走价格
    frames.append(pl.DataFrame({
        "symbol": sym,
        "ts": ts,
        "open": close + rng.normal(0, 0.2, n),
        "high": close + np.abs(rng.normal(0, 0.5, n)),
        "low": close - np.abs(rng.normal(0, 0.5, n)),
        "close": close,
        "volume": rng.integers(1_000, 50_000, n),
    }))
pl.concat(frames).write_parquet("bars.parquet")

sig_frames = []
for sym in ["AAPL", "MSFT"]:
    idx = rng.choice(len(ts), 80, replace=False)        # 每个标的 80 笔随机信号
    sig_frames.append(pl.DataFrame({
        "symbol": sym,
        "ts": ts[idx],
        "side": rng.choice(["BUY", "SELL"], 80),
        "qty": rng.integers(10, 200, 80),
    }))
pl.concat(sig_frames).write_parquet("signals.parquet")

pl.DataFrame({
    "symbol": ["AAPL", "MSFT"],
    "name": ["Apple Inc.", "Microsoft Corp."],
    "sector": ["Tech", "Tech"],
}).write_parquet("symbols.parquet")
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
          symbols.with_columns(pl.col("sector").cast(pl.Categorical)),
          on="symbol", how="left",
      )
)
```

## 15.3 rolling 指标与信号计算

```python
daily = (
    pl.scan_parquet("bars.parquet")
    .with_columns(
        ma_fast=pl.col("close").rolling_mean(20).over("symbol"),
        ma_slow=pl.col("close").rolling_mean(60).over("symbol"),
        # ATR 波动率：标准 True Range = 三项取 max，再滚动平均
        atr=pl.max_horizontal(
              pl.col("high") - pl.col("low"),
              (pl.col("high") - pl.col("close").shift(1)).abs(),
              (pl.col("low") - pl.col("close").shift(1)).abs(),
             ).rolling_mean(14).over("symbol"),
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
    )
    .with_columns(
        # 滚动夏普：ret 定义在上一个 with_columns 中，同层引用会报 ColumnNotFoundError
        rolling_sharpe_30d=pl.col("ret").rolling_mean(30) / pl.col("ret").rolling_std(30),
    )
    # 最大回撤：累计收益的全程峰值（cum_max）与当前值之差
    .with_columns(cum=pl.col("daily_pnl").cum_sum())
    .with_columns(
        drawdown=pl.col("cum") - pl.col("cum").cum_max()
    )
    .collect()
)
```

三个统计口径的正确姿势：

- **滚动夏普**用 `rolling_mean / rolling_std`（同窗口口径），不要用 `over(int_range // N)` 之类近似——分块均值混着不同窗口的数据
- **最大回撤**用 `cum_sum - cum_max().shift(1)` 变体可排除"当日创新高"的零回撤干扰；示例简化为 `cum - cum_max`
- **年化**：`sharpe * sqrt(252)`（日线）、`sqrt(252 * 24 * 60)`（分钟线）——窗口口径与年化因子必须匹配

## 15.5 信号驱动回测：从金叉到权益曲线

15.3 的 `golden_cross` 不该止步于一张布尔表——完整的回测要把它变成**持仓**，再变成**权益曲线**。这依然是纯向量化表达式：

```python
equity = (
    daily
    # 死叉信号：金叉的镜像（金叉开仓、死叉平仓）
    .with_columns(
        death_cross=(pl.col("ma_fast") < pl.col("ma_slow"))
                     & (pl.col("ma_fast").shift(1) >= pl.col("ma_slow").shift(1)),
    )
    # 信号 → 持仓：金叉置 1、死叉置 0、其余 forward_fill 继承前一状态
    .with_columns(
        position=pl.when(pl.col("golden_cross")).then(1)
                    .when(pl.col("death_cross")).then(0)
                    .otherwise(None)
                    .forward_fill()
                    .fill_null(0)        # 建仓前的空仓期
                    .over("symbol"),      # 每个标的独立持仓状态
    )
    # 持仓收益 = 前一日持仓 × 当日涨跌幅（T 日收盘出信号，T+1 日才持仓）
    .with_columns(
        daily_ret=pl.col("close").pct_change(),
    )
    .with_columns(
        strat_ret=pl.col("daily_ret") * pl.col("position").shift(1).fill_null(0),
    )
    # 权益曲线与回撤
    .with_columns(
        equity=pl.col("strat_ret").cum_sum().over("symbol"),
        drawdown=(pl.col("strat_ret").cum_sum()
                  - pl.col("strat_ret").cum_sum().cum_max()).over("symbol"),
    )
    .collect()
)

print(equity.filter(pl.col("symbol") == "AAPL")
            .select("ts", "position", "daily_ret", "strat_ret", "drawdown")
            .tail(5))
print("AAPL 金叉次数:", equity.filter(pl.col("symbol") == "AAPL")
                               .get_column("golden_cross").sum())
```

关键机制拆解：

1. **`when/then/otherwise(None)` + `forward_fill`** 是"信号驱动的状态机"惯用法——只在信号日改状态，其余日期继承，全程无 Python 循环
2. **`over("symbol")`** 保证每个标的的持仓状态独立演化——漏掉它，标的 A 的金叉会"串仓"到标的 B
3. **`position.shift(1)`** 避免"未来函数"：当日信号只能驱动次日的收益，这是回测最常见的正确性陷阱

> 15.4 的"逐笔成交"与 15.5 的"信号驱动持仓"是两种回测范式：前者回答"这批信号执行得如何"，后者回答"这个策略（规则）表现如何"。生产中通常先跑 15.5 验证策略，再用 15.4 评估执行。

## 15.6 剖析：over 的成本与收益

```python
# over("symbol") 的分区数 = 证券数量（如 5000）
# 每个分区内 rolling 串行（窗口有行间依赖），分区间并行
# 对比：错误写法（无 over）会把所有证券的 K 线混在一个窗口里
```

分区并行的收益可以直接实测（50 万行 × 50 个标的）：

```python
import time
import numpy as np

rng = np.random.default_rng(5)
big = pl.select(
    symbol=pl.int_range(0, 500_000) % 50,
    ts=pl.int_range(0, 500_000),
    close=pl.Series(rng.normal(100, 5, 500_000)),
).sort(["symbol", "ts"])

t0 = time.perf_counter()
big.with_columns(ma=pl.col("close").rolling_mean(20).over("symbol"))
t1 = time.perf_counter()
big.with_columns(ma=pl.col("close").rolling_mean(20))   # 故意错：混窗口
t2 = time.perf_counter()
print(f"over 50 分区: {t1 - t0:.3f}s | 无 over: {t2 - t1:.3f}s")
# 注意：无 over 反而更慢——它被迫把 50 个标的的数据排进单个窗口逻辑
# 正确的对照是"单分区串行"：分区并行让 rolling 吞吐接近线性扩展
```

```python
# set_sorted 是向引擎"承诺"有序：跳过有序性检查，省一次全表排序
# 若数据实际无序而强行声明，join_asof 会静默给出错误匹配（见练习 2）
```

什么时候 over 会变慢：分区数远大于核数且每分区只有几行时（如 `over("user_id")` 而 user 有百万级）——调度开销超过计算本身。此时考虑 `group_by` 聚合或先 `sort` 再 `over`（有序数据引擎可用更快路径）。

## 15.7 避免逐标的循环

```python
# ❌ 反模式：5000 个标的循环回测（示意，勿运行）
# symbols_list = bars.select("symbol").unique()["symbol"]  # 5000 个标的
# for symbol in symbols_list:
#     df_sym = bars.filter(pl.col("symbol") == symbol)     # 5000 次全表扫描
#     run_backtest(df_sym)                                 # 每个标的从头执行一遍管道

# ✅ 向量化：一次扫描，over 分区并行处理全部标的
bars.with_columns(
    pl.col("close").rolling_mean(20).over("symbol")
).collect().head(3)
# 提速典型值：几十倍——不仅省了循环，还让优化器看到全貌
```

## 要点回顾

- `join_asof` 是事件对齐行情的标准工具（先排序）
- 信号 = 布尔表达式，持仓 = `when/then + forward_fill` 状态机，回测 = 分组聚合——全程向量化
- `position.shift(1)` 防未来函数：当日信号驱动次日收益
- `over("symbol")` 分区并行替代外层循环；分区数远大于核数时反而是反模式
- 策略回测（15.5）与执行评估（15.4）是两种范式，先验证策略再评估执行

## 性能检查清单

- [ ] 时间列已 `set_sorted`？（join_asof 硬性要求）
- [ ] 是否消除了逐标的 Python 循环？
- [ ] 主数据低基数列是否 cast 成 Categorical？
- [ ] rolling 窗口的行间依赖是否限制了并行度（窗口越短越好）？
- [ ] 信号→收益链路是否有 `shift(1)` 防未来函数？
- [ ] `over` 的分区数是否在"远小于行数、不远大于核数"的甜蜜区？

## 练习

1. **金叉验证**：构造单调上升后转跌的价格序列，用 15.3 节表达式计算 golden_cross，验证信号恰好在均线交叉的那一根 K 线为 True。
2. **排序前提**：把 `bars` 的行顺序打乱（如 `bars.collect().sample(fraction=1.0, shuffle=True)`）后直接 `join_asof`，观察引擎并不报错、但 close 对齐结果悄悄错乱；再 `sort("ts").set_sorted("ts")` 后重算，对比两种结果——体会 `set_sorted` 是"承诺"而非"保证"。
3. **回撤计算**：构造一条先涨后跌的收益曲线，用 15.4 节的 `cum_sum + cum_max` 计算最大回撤，手工验证峰值谷值。
4. **状态机扩展**：给 15.5 节的持仓逻辑加一个止损规则——当回撤超过 5% 时强制平仓（提示：`pl.when(drawdown < -0.05).then(0).otherwise(None).forward_fill()` 与金叉/死叉信号组合时，注意规则优先级）。
5. **年化夏普**：把 15.4 节的 `rolling_sharpe_30d` 乘上 `sqrt(252)` 年化，对比 15.5 节权益曲线的全程夏普，讨论滚动与全程口径的适用场景。
