"""示例 2：链式调用与流式管道（对应第 3、5、6、8 章）

运行：uv run python examples/02_lazy_streaming.py
"""
from datetime import datetime
from pathlib import Path

import polars as pl

WORK = Path("example_data")
WORK.mkdir(exist_ok=True)

# --- 第 3 章：生成分区 Parquet 数据 ---
for day in range(1, 4):
    n = 100_000
    part_dir = WORK / f"date=2026-08-{day:02d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    (
        pl.DataFrame({
            "ts": pl.datetime_range(
                datetime(2026, 8, day), datetime(2026, 8, day, 23, 59),
                "1m", eager=True,
            ).sample(n, with_replacement=True, seed=day),
            "level": pl.Series(["ERROR", "INFO", "WARN"] * n)[:n],
            "endpoint": pl.Series(["/api/a", "/api/b", "/api/c", "/api/d"] * n)[:n],
            "latency_ms": pl.Series(range(n)) % 500 + day * 10,
        })
        .write_parquet(WORK / f"date=2026-08-{day:02d}" / "part-0.parquet")
    )

# --- 第 6 章：explain 观察谓词下推与投影裁剪 ---
lf = (
    pl.scan_parquet(WORK / "date=2026-08-*" / "*.parquet")
      .filter(pl.col("level") == "ERROR")
      .select(["endpoint", "latency_ms"])
)
print("--- explain（默认优化）---")
print(lf.explain())
print("--- explain（关闭优化）---")
print(lf.explain(optimizations=pl.QueryOptFlags.none()))

# --- 第 8 章：端到端流式管道（scan → 变换 → sink）---
out = WORK / "error_stats.parquet"
(
    pl.scan_parquet(WORK / "date=2026-08-*" / "*.parquet")
      .filter(pl.col("level") == "ERROR")
      .with_columns(hour=pl.col("ts").dt.truncate("1h"))
      .group_by(["hour", "endpoint"])
      .agg(
          pl.len().alias("n"),
          pl.col("latency_ms").mean().alias("avg_latency"),
          pl.col("latency_ms").quantile(0.99).alias("p99_latency"),
      )
      .sink_parquet(out)
)
stats = pl.read_parquet(out)
print(f"\n流式管道产出: {stats.height} 行（3 天 × 24 小时 × 4 端点 = 288）")
print(stats.sort("hour", "endpoint").head(6))

# --- 第 5 章：链式 vs 逐步赋值（计时对比；profile 自 1.43 起弃用，见 12.1 的替代工具）---
import time

df_eager = pl.read_parquet(WORK / "date=2026-08-*" / "*.parquet")
t0 = time.perf_counter()
step = df_eager.filter(pl.col("level") == "ERROR").select(pl.col("latency_ms").mean())
t_step = time.perf_counter() - t0

lf_mean = lf.select(pl.col("latency_ms").mean())
lf_mean.collect()                       # 预热
t0 = time.perf_counter()
lf_mean.collect()
t_lazy = time.perf_counter() - t0
print(f"\n逐步赋值: {t_step*1e3:.1f} ms vs 惰性链式: {t_lazy*1e3:.1f} ms")

# 清理示例数据
import shutil
shutil.rmtree(WORK)
print("\n示例数据已清理")
