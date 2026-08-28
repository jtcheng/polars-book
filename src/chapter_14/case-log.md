# 第 14 章 实战：亿级日志分析管道

> 本章要解决什么问题：综合流式主线（scan → 变换 → sink）处理亿级行日志，全程内存受控。

## 14.1 场景与数据

- 输入：多日期分区的 Parquet 日志（如 200GB、10 亿行）
- 输出：小时级错误统计 + 慢端点 TopN

```python
# 数据形态示例
# logs/date=2026-08-27/part-000.parquet
# logs/date=2026-08-28/part-000.parquet
# 列：ts (Datetime), level (String), endpoint (String),
#     latency_ms (Int64), user_id (Int64)
```

先运行一次数据生成（2 个日期分区 × 2000 行日志），本章代码即可顺序执行：

```python
import polars as pl
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np

rng = np.random.default_rng(11)
for day in [date(2026, 8, 27), date(2026, 8, 28)]:
    part = Path(f"logs/date={day.isoformat()}")
    part.mkdir(parents=True, exist_ok=True)
    n = 2000
    pl.DataFrame({
        "ts": [datetime(2026, 8, day.day) + timedelta(minutes=int(i))
               for i in rng.integers(0, 1440, n)],
        "level": rng.choice(["INFO"] * 6 + ["WARN"] * 2 + ["ERROR"] * 2, n),
        "endpoint": rng.choice(["/api/a", "/api/b", "/api/c", "/api/d"], n),
        "latency_ms": rng.integers(5, 2000, n),
        "user_id": rng.integers(1, 100, n),
    }).sort("ts").write_parquet(part / "part-000.parquet")
```

## 14.2 端到端流式管道

```python
import polars as pl

(pl.scan_parquet("logs/date=2026-08-*/*.parquet")
   .filter(pl.col("level") == "ERROR")                     # 谓词下推
   .with_columns(hour=pl.col("ts").dt.truncate("1h"))     # 时间粒度
   .group_by(["hour", "endpoint"])
   .agg(
       pl.len().alias("n"),
       pl.col("latency_ms").mean().alias("avg_latency"),
       pl.col("latency_ms").quantile(0.99).alias("p99_latency"),  # 百分位监控
   )
   .sink_parquet("error_stats.parquet"))                    # 流式写出
```

内存分析：管道的可变工作集 = 单个 morsel（几十万行）+ 增量哈希表（720 小时 × ~200 端点 ≈ 14 万键）。**输入 200GB，内存占用稳定在几十 MB**。

## 14.3 剖析

### 谓词下推如何跳过无 ERROR 的行组

```python
# 验证：计划中的 SELECTION 确实下推到扫描层
lf = (pl.scan_parquet("logs/date=2026-08-*/*.parquet")
        .filter(pl.col("level") == "ERROR"))
print(lf.explain())
# Parquet SCAN 中出现 SELECTION —— 存储层直接跳过 level 列
# 不含 ERROR 的行组（统计信息可证明）整块跳过，读取量骤减
```

### TopN 的两阶段：先聚合再 sort

```python
# ❌ 错误直觉：对原始数据 sort 再取头（10 亿行全量排序，示意勿运行）
# big.sort("latency_ms", descending=True).head(100)

# ✅ 正确策略：先聚合（收窄到端点粒度），再排序
(pl.scan_parquet("logs/date=2026-08-*/*.parquet")
   .group_by("endpoint")
   .agg(pl.col("latency_ms").mean().alias("avg_latency"))
   .collect()
   .sort("avg_latency", descending=True)
   .head(20))
# 聚合后只有 ~200 行，sort 成本可忽略
```

## 14.4 增量运行与监控

### 增量运行：只处理新增分区

```python
# anti join 去重：排除已入库的小时（幂等保证）
already = pl.scan_parquet("error_stats.parquet").select("hour", "endpoint")

(pl.scan_parquet("logs/date=2026-08-2*/*.parquet")         # glob 收窄到新增分区
   .filter(pl.col("level") == "ERROR")
   .with_columns(hour=pl.col("ts").dt.truncate("1h"))
   .group_by(["hour", "endpoint"]).agg(pl.len().alias("n"))
   .join(already, on=["hour", "endpoint"], how="anti")     # 增量去重
   .sink_parquet("error_stats_increment.parquet"))
```

### 与 DuckDB 协作做 ad-hoc SQL

```python
import duckdb

# 管道产出的 Parquet 直接用 SQL 探索——无需再次加载
duckdb.sql("""
    SELECT endpoint, p99_latency
    FROM 'error_stats.parquet'
    WHERE p99_latency > 1000
    ORDER BY p99_latency DESC
""").show()
```

## 14.5 环比监控（呼应第 11 章）

```python
(pl.scan_parquet("error_stats.parquet")
   .sort("hour")
   .with_columns(
       mom=pl.col("n").pct_change(24).over("endpoint"),  # 与 24 小时前比较
   )
   .filter(pl.col("mom") > 0.5)   # 错误量环比暴涨 50% 的端点
   .collect())
```

## 要点回顾

- 流式管道的内存上界 = morsel + 哈希表（基数决定）
- TopN 永远先聚合再排序
- anti join 是幂等增量处理的利器

## 性能检查清单

- [ ] 峰值内存是否稳定在预期上界？（用 `/usr/bin/time -l` 实测）
- [ ] 实测峰值内存验证流式效果？（1.41+ 新引擎 `explain` 无 STREAMING 标记，见第 8 章）
- [ ] 聚合基数（小时 × 端点）是否预估过？
- [ ] 增量运行是否幂等（anti join 保障）？

## 练习

1. **模拟数据**：用 `pl.datetime_range` + 随机数生成分区日志（3 天 × 4 端点 × 10 万行/天），写入 `logs/date=*/part-0.parquet`，运行 14.2 节管道并验证输出行数 = 3 天 × 4 端点。
2. **p99 对比**：把聚合中的 `quantile(0.99)` 分别换成 `max` 与 `mean`，三种口径下找出"最慢端点"是否一致？讨论监控该用哪个。
3. **幂等验证**：连续运行两次 14.4 节的增量管道（anti join 版本），验证第二次输出的增量为 0 行。
