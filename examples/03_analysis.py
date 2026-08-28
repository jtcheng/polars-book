"""示例 3：聚合、join 与时间序列（对应第 10、11 章）

运行：uv run python examples/03_analysis.py
"""
from datetime import datetime

import polars as pl

# --- 第 10 章：join 全家桶 ---
orders = pl.DataFrame({
    "user_id": [1, 2, 3, 4],
    "amount": [100, 200, 300, 400],
})
users = pl.DataFrame({
    "user_id": [1, 2, 5],
    "name": ["a", "b", "e"],
})

inner = orders.join(users, on="user_id", how="inner")
semi = orders.join(users.select("user_id"), on="user_id", how="semi")
anti = orders.join(users.select("user_id"), on="user_id", how="anti")
print(f"inner: {inner.height} 行 | semi: {semi.height} 行 | anti: {anti.height} 行")
assert inner.height == 2 and semi.height == 2 and anti.height == 2

# --- 第 10 章：IQR 异常检测 ---
df = pl.DataFrame({
    "amount": [10.0, 12.0, 11.0, 13.0, 12.5, 11.8, 1000.0],  # 末行是异常值
})
cleaned = (
    df.with_columns(
        q1=pl.col("amount").quantile(0.25, interpolation="linear"),
        q3=pl.col("amount").quantile(0.75, interpolation="linear"),
    )
    .with_columns(iqr=pl.col("q3") - pl.col("q1"))
    .filter(pl.col("amount").is_between(
        pl.col("q1") - 1.5 * pl.col("iqr"),
        pl.col("q3") + 1.5 * pl.col("iqr"),
    ))
)
print(f"IQR 剔除异常: {df.height} → {cleaned.height} 行")
assert cleaned.height == 6

# --- 第 10 章：join_asof 时序对齐 ---
trades = pl.DataFrame({"ts": [1, 5, 9], "price": [10.0, 11.0, 12.0]}).set_sorted("ts")
quotes = pl.DataFrame({"ts": [2, 6, 10], "bid": [9.5, 10.5, 11.5]}).set_sorted("ts")
asof = trades.join_asof(quotes, on="ts", strategy="backward")
print("join_asof:", asof.to_dicts())

# --- 第 11 章：重采样 OHLC ---
bars = pl.DataFrame({
    "ts": pl.datetime_range(datetime(2026, 8, 1, 9), datetime(2026, 8, 1, 9, 59), "1m", eager=True),
    "price": 100 + pl.Series(range(60), dtype=pl.Float64) * 0.1,
})
ohlc = (
    bars.group_by_dynamic("ts", every="15m", closed="left")
        .agg(
            open=pl.col("price").first(),
            high=pl.col("price").max(),
            low=pl.col("price").min(),
            close=pl.col("price").last(),
        )
        .sort("ts")
)
print(ohlc)

# --- 第 11 章：环比 + 缺失周期陷阱 ---
sparse = pl.DataFrame({
    "month": ["2026-01", "2026-03"],   # 缺 2 月
    "total": [100, 130],
}).with_columns(pl.col("month").str.to_date("%Y-%m"))

wrong = sparse.with_columns(mom=pl.col("total").pct_change(1))
print(f"错误环比（跳过 2 月）: {wrong['mom'][1]:.2f}")  # 0.3——不是环比

dense = sparse.upsample("month", every="1mo").with_columns(
    pl.col("total").forward_fill()
)
right = dense.with_columns(mom=pl.col("total").pct_change(1))
print(f"补齐后逐月环比: 2 月 {right['mom'][1]:.2f}, 3 月 {right['mom'][2]:.2f}")
