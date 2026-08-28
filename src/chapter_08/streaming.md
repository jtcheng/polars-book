# 第 8 章 流式操作与流式执行引擎

> 本章要解决什么问题：掌握 scan → 惰性变换 → sink 的端到端流式管道，理解 new streaming engine 的 morsel 机制，知道哪些操作可流式、哪些会断流。这是全书流式主线的核心章。

## 8.1 端到端流式管道

```mermaid
flowchart LR
    A["scan_parquet<br/>（惰性打开）"] --> B["filter<br/>（谓词下推）"]
    B --> C["with_columns<br/>（变换）"]
    C --> D["group_by().agg<br/>（增量聚合）"]
    D --> E["sink_parquet<br/>（流式写出）"]
    style A fill:#e1f5fe
    style E fill:#e8f5e9
```

```python
(pl.scan_parquet("logs/*.parquet")
   .filter(pl.col("level") == "ERROR")
   .with_columns(hour=pl.col("ts").dt.truncate("1h"))
   .group_by("hour")
   .agg(pl.len().alias("n"))
   .sink_parquet("error_stats.parquet"))  # 全程内存受控
```

这条管道即使输入是 200GB，内存占用也只有：单个 morsel 的工作集 + 增量聚合的哈希表（小时数 × 列宽）。**内存复杂度 O(基数) 而非 O(数据量)**。

### collect vs sink 的内存对比

```python
import os
import polars as pl

# ❌ collect 路径：全量物化到内存再写盘
df = (pl.scan_parquet("logs/*.parquet")
        .group_by("endpoint").agg(pl.len()).collect())
df.write_parquet("out.parquet")

# ✅ sink 路径：不物化，逐 morsel 流过引擎
(pl.scan_parquet("logs/*.parquet")
   .group_by("endpoint").agg(pl.len())
   .sink_parquet("out.parquet"))
# 用 /usr/bin/time -l 或资源监视器对比两者峰值 RSS
```

## 8.2 new streaming engine

- 新流式引擎（1.41+ 进入 stable）取代了旧流式引擎：旧引擎只让部分节点流式、靠 `explain` 的 STREAMING 标记识别；新引擎整条管道默认以 morsel 为单元流式执行
- `collect(engine="streaming")` 显式选用新引擎（默认 `engine="auto"`，2.0 起流式将成为默认引擎）
- morsel 机制：分块读取 → 分块处理 → 分块写出
- 峰值内存控制原理

```python
# 流式 collect：结果仍是 DataFrame，但中间过程流式
result = (
    pl.scan_parquet("logs/*.parquet")
      .filter(pl.col("amount") > 0)
      .group_by("user_id")
      .agg(pl.col("amount").sum())
      .collect(engine="streaming")   # 显式指定流式引擎
)
# 适用：结果集小（聚合结果），但中间数据巨大
```

```python
# 异步 sink：sink 本身也可以不阻塞主线程
sink_lf = (
    pl.scan_parquet("logs/*.parquet")
      .group_by("endpoint").agg(pl.len())
      .sink_parquet("out.parquet", lazy=True)   # 只登记计划，不执行
)
# ...此处可继续登记其它 sink，统一调度...
sink_lf.collect()   # 此刻才真正执行；返回空 DataFrame，数据已在 out.parquet
```

## 8.3 哪些操作可流式

```mermaid
flowchart TD
    OP{"操作类型"} -->|"filter / select / with_columns<br/>（逐行无状态）"| YES["天然流式<br/>逐 morsel 处理"]
    OP -->|"group_by.agg<br/>（有限基数）"| YES2["流式<br/>增量哈希聚合"]
    OP -->|"sort / unique<br/>（全序依赖）"| OOC["流式（out-of-core）<br/>1.42+ 内存不足时溢写磁盘<br/>内存充足时退回全内存更快"]
    OP -->|"join 大表"| OOC2["流式（out-of-core）<br/>1.42+ 同上"]
```

新引擎下 sort/join 支持 out-of-core：内存不足时把中间数据溢写磁盘、逐块归并，代价是 I/O；内存充足时仍走全内存路径，速度更快。仍有少数复杂算子（如某些跨块全局窗口）可能退回全内存执行，以实测为准。

### 验证手段：实测峰值内存

旧验证方法 `explain(streaming=True)` 已于 1.25 弃用，且 1.41+ 新流式引擎的 `explain` 输出**不再包含 STREAMING 节点标记**——整条管道默认就在流式，计划文本看不出差别。可靠的验证手段是实测峰值 RSS：管道真的在流式，峰值内存就不随数据量线性增长。

```python
# 用子进程跑管道，resource 读取峰值 RSS（跨平台，无需外部工具）
import subprocess
import sys
import resource

CHILD = """
import polars as pl
(pl.scan_parquet("logs/*.parquet")
   .filter(pl.col("level") == "ERROR")
   .with_columns(hour=pl.col("ts").dt.truncate("1h"))
   .group_by("hour")
   .agg(pl.len().alias("n"))
   .sink_parquet("error_stats.parquet"))
"""

proc = subprocess.run([sys.executable, "-c", CHILD])
assert proc.returncode == 0
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if sys.platform == "linux":       # macOS 返回字节，Linux 返回千字节
    peak *= 1024
print(f"峰值 RSS ≈ {peak / 1024 / 1024:.0f} MB")
```

```bash
# 命令行一行搞定
/usr/bin/time -l python pipeline.py   # macOS：看 "peak memory footprint"
/usr/bin/time -v python pipeline.py   # Linux：看 "Maximum resident set size"
```

把输入规模放大 5 倍再跑一次：峰值 RSS 几乎不变 → 内存复杂度 O(基数) 而非 O(数据量)，这就是流式执行的直接证据。

## 8.4 与 batch 处理结合

- 按 partition 列分批 scan
- 外层循环 + `pl.concat` 增量合并

```python
# 流式不适用时的降级策略：按分区批处理
from datetime import date, timedelta

results = []
d = date(2026, 8, 1)
while d < date(2026, 8, 28):
    lf = (
        pl.scan_parquet(f"logs/date={d.isoformat()}/*.parquet")
          .group_by("endpoint").agg(pl.len())   # 单日基数小
    )
    results.append(lf.collect())
    d += timedelta(days=1)

# 二次归并：批结果再聚合
(pl.concat(results)
   .group_by("endpoint").agg(pl.col("len").sum())
   .sort("len", descending=True))
```

## 8.5 流式的边界：基数决定一切

```python
# group_by 可流式的前提：唯一键数量可控
# ✅ 基数 = 小时数 × 端点数 ≈ 720 × 200 = 14 万 → 哈希表 ~MB 级
lf.group_by(["hour", "endpoint"]).agg(pl.len())

# ❌ 基数 = user_id 数 = 2 亿 → 哈希表爆炸，"流式"名存实亡
lf.group_by("user_id").agg(pl.len())
# 对策：按 user_id 哈希分片，分多次流式处理
```

## 要点回顾

- scan → 变换 → sink 全程内存 O(分块) 而非 O(全量)
- 新流式引擎（1.41+）整条管道默认以 morsel 为单元流式执行；验证靠实测峰值 RSS，而非旧的 STREAMING 标记
- 聚合基数是流式可行性的决定因素

## 性能检查清单

- [ ] 是否用 sink_* 替代了 collect 后再 to_parquet？
- [ ] group_by 的基数（unique key 数）是否可控？
- [ ] 是否实测过峰值 RSS，确认内存不随数据量线性增长？
- [ ] sort/join 大表是否留意过 out-of-core 路径的内存/速度取舍？
- [ ] 峰值内存是否实测过（time -l / subprocess + resource）？

## 练习

1. **规模不变性**：对同一条 sink 聚合管道，分别输入 1GB 与 5GB 数据（可循环 sink 生成），用 `subprocess` + `resource.getrusage` 记录两次峰值 RSS。若两者接近，就证明了内存复杂度 O(基数) 而非 O(数据量)——这是新流式引擎（1.41+）替代旧 STREAMING 标记的验证方法。
2. **内存实测**：生成 5GB 分区 Parquet（可用循环 sink），分别以 collect 和 sink 两种方式聚合，用 `/usr/bin/time -l`（macOS）或 `/usr/bin/time -v`（Linux）记录峰值 RSS 差异。
3. **基数实验**：对同一份大数据分别按低基列（如 weekday）与高基列（如 user_id）流式聚合，观察内存曲线差异，验证"基数决定一切"。
