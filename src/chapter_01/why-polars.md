# 第 1 章 为什么是 Polars

> 本章要解决什么问题：理解 Polars 在数据处理工具生态中的定位，以及"单机多核时代回归"的背景，为后续章节建立动机。

## 1.1 单机多核时代的回归

- 摩尔定律转向多核；分布式不是唯一答案
- 内存带宽 vs 磁盘/网络带宽：单机列式引擎的优势数量级
- 什么规模以下，单机引擎是最优解

"单机够用"不是情怀，是三个带宽数量级撑起来的事实。先看硬件的现实：现代桌面/工作站 CPU 的内存带宽约 **50~100 GB/s**，NVMe SSD 约 **3~7 GB/s**，千兆网络仅 **0.1 GB/s**——三者相差两到三个数量级。这意味着：数据在本机内存里流转时，引擎的多核算力可以喂得饱；一旦跨进程边界走网络，带宽瞬间塌缩三个数量级，再多的核也白搭。摩尔定律的另一个转向同样关键：单核主频十几年前就停在 3~4 GHz，晶体管密度的增量全部投向了核数——今天的"一台电脑"本质是一台共享内存的并行计算机，而 pandas 时代的工具链是按单线程设计的，多核算力被闲置。列式单机引擎正是踩在这两个交汇点上：列式布局让顺序扫描吃满内存带宽，多线程让哈希聚合与 join 用满所有核。

规模判据由此可估：设数据解压后为 S、聚合基数为 B，则单机内存管道需要 S ≤ RAM（可加磁盘交换，见第 8 章流式）；流式管道（scan → sink）进一步把要求降到 B 与单个 morsel 能装下即可，S 只受磁盘容量约束——几百 GB 的数据在单机 NVMe 上完全可行。跨过 TB、或需要多租户共享资源时，分布式的账才真正开始划算。

先安装并确认环境：

```python
# pip install polars   # 或: uv add polars
import polars as pl

print(pl.__version__)          # 本书基于 1.x
print(pl.thread_pool_size())    # 查看可用并行度，例如 10
```

一个直观感受——你的机器就是一台并行计算机：

```python
# 生成 1000 万行示例数据，观察多线程聚合的耗时
import time

# 注意：pl.DataFrame 不接受 Expr，用 pl.select 求值表达式生成列
df = pl.select(
    user=pl.int_range(0, 10_000_000, dtype=pl.Int64) % 1_000,
    amount=pl.int_range(0, 10_000_000, dtype=pl.Int64) % 10_000,
)

t0 = time.perf_counter()
result = df.group_by("user").agg(pl.col("amount").sum())
print(f"group_by 聚合耗时: {time.perf_counter() - t0:.3f}s")  # 典型值：几十毫秒
```

同样的操作如果用 Python 循环逐行累加，通常需要数秒——差距来自列式内存与多线程（第 2、7 章展开）。

## 1.2 生态定位对比

```mermaid
flowchart TB
    subgraph loc["数据规模"]
        L1["MB ~ 数 GB"]
        L2["数 GB ~ 数百 GB"]
        L3["TB 以上 / 多租户"]
    end
    pandas["pandas<br/>单线程 · Eager · 灵活但慢"]
    polars["Polars<br/>多线程 · Lazy · 单机高性能"]
    duckdb["DuckDB<br/>SQL 优先 · OLAP 查询"]
    spark["Spark<br/>分布式 · 运维重"]
    pandas --> L1
    polars --> L2
    duckdb --> L2
    spark --> L3
```

- **vs pandas**：列式内存、多线程、表达式引擎；谁在迁移成本上占优
- **vs Spark**：免运维、无序列化开销、迭代交互式分析体验
- **vs DuckDB**：表达式 API vs SQL；两者其实可以协作（第 13、16 章）

三笔账分开算。从 pandas 迁过来，真正的成本不在语法而在执行模型的心智：pandas 每一步都立即执行、随时可以打印中间结果排查问题，Polars 的 `scan_*` 返回 LazyFrame、先把计算描述完、collect 时才一次执行（DataFrame API 本身仍是 eager 的，第 6 章辨析两种模式），排查方式从"看每一步的输出"变成"看计划与最终结果"。好在概念层两者几乎一一对应——select/filter/group_by/join 的语义可以直接平移——所以短脚本近乎机械改写，下一小节就是直观对照；真正要重训的只有"先写完再执行"这一条习惯，管道越长越值得先按第 6 章的方法梳理数据流再动手。

对 Spark 的账要算在分布式本身的固定开销上：数据在 executor 之间流动要经网络与序列化，每次 shuffle 都是一笔分区级的大批量传输与编解码税，而单机引擎的数据始终驻留在同一块内存里，线程之间只传递数据分片的引用，这笔税根本不存在。运维上同样不对称：Spark 需要 JVM、集群管理与多组件版本协调，Polars 只是随脚本安装的一个 pip 依赖。对"改一版口径就重跑"的迭代式分析，每轮作业的提交与预热延迟会被反复放大，这正是两类工具体验差距的主要来源。

DuckDB 与 Polars 的关系是分工而非竞争：前者以 SQL 为一等公民，适合分析师、BI 工具直连与存量 SQL 资产的复用；后者以表达式 API 为核心，适合列名与过滤条件都来自变量的程序化管道——同样的动态逻辑换成字符串拼 SQL，既没有类型检查也不好维护。两者之间有 Arrow 这座桥，查询结果可以零拷贝在两边流转，完全可以在一条管道里让 SQL 做声明式查询、表达式做后续变换（第 13 章讲互通细节，第 16 章实战两者同台）。

### 同一任务的两种写法

统计每个用户的首单金额与订单数。pandas 需要多次物化中间结果：

```python
# pandas：三步三次物化
import pandas as pd

pdf = pd.read_csv("orders.csv")
first_orders = pdf.sort_values("ts").groupby("user_id").first()   # 注意：见下方说明
counts = pdf.groupby("user_id")["order_id"].count()
result = first_orders.join(counts, rsuffix="_count")
```

> 一个语义细节：pandas 的 `groupby().first()` 会**跳过组内缺失值**（首行 amount 为 NaN 时取下一个非空值），而下面 Polars 版的 `sort_by("ts").first()` 取的是 ts 最小那一行的 amount（是 null 就是 null）。要对齐语义，pandas 侧应改用 `.nth(0)`。迁移时这类"名字相同、细节不同"的地方最值得逐条核对。

Polars 一条链式表达，一次执行：

```python
# polars：一次构建、整体优化（第 5、6 章展开）
(pl.scan_csv("orders.csv")              # 惰性扫描，不立即读取
   .group_by("user_id")
   .agg(
       pl.col("amount").sort_by("ts").first().alias("first_amount"),
       pl.len().alias("n_orders"),
   )
   .collect()
)
```

注意：`scan_csv` 直到 `collect()` 才真正读取数据——这就是惰性执行（第 6 章）。

## 1.3 性能基准概览

- 典型工作负载（group_by/join/字符串处理）的基准对比
- 注明 Polars 版本与数据规模

选 group_by、join、字符串处理这三类负载，是因为它们各自压着引擎的一条主轴：group_by 压哈希聚合的多线程伸缩，join 压内存布局与探测效率，字符串处理则暴露计算有没有绕道 Python 解释器——三者差距的来源不同，合起来才构成一张完整的性能画像。因此读基准时盯量级差距而非绝对秒数：具体耗时属于特定环境，"哪类操作贵、贵在哪个环节"的结构信息才是可迁移的结论。

你可以复现的最小基准——1 亿行数值聚合：

```python
import time
import polars as pl

N = 100_000_000
# pl.select 求值表达式生成列，再转 LazyFrame
lf = (
    pl.select(k=pl.int_range(0, N, dtype=pl.Int64) % 1_000,
              x=pl.int_range(0, N, dtype=pl.Int64) % 1_000_000)
    .lazy()
)

t0 = time.perf_counter()
result = lf.group_by("k").agg(pl.col("x").sum()).collect()
print(f"1 亿行 group_by: {time.perf_counter() - t0:.2f}s")
# 基准环境：Apple M2 Pro (10 核)，polars 1.44.1，典型值 ~0.5s
```

把 `pl.col("x").sum()` 换成 `pl.col("x").cast(pl.String)` 再试一次，就能体会数值与字符串处理的数量级差异（注意：两者输出形态不同——`sum` 每键产一个标量，`cast` 每键物化一个字符串列表，耗时差异同时来自类型转换与列表构建；练习 2 给了控制变量的对比方式）。join 负载的基准见第 10 章（哈希 join 的实测），字符串处理见第 9 章。

> **基准测试提醒**：任何性能数字都与硬件、版本、数据分布强相关。本书所有基准均注明环境，且建议你用自己的负载复测（第 12 章系统化讲解基准方法论）。

## 要点回顾

- Polars 填补了 pandas 与 Spark 之间的单机高性能空档
- 列式内存 + 多线程 + 惰性优化是三大性能支柱
- `scan_*` + 链式表达 + `collect()` 是本书反复出现的基本形态

## 性能检查清单

在选型或动手优化前，先回答这些问题：

- [ ] 我的数据规模单机内存（含磁盘交换）能否容纳？——数 GB 到数百 GB 是 Polars 甜蜜区
- [ ] 瓶颈是计算还是 I/O？——`explain()` 先看计划，不靠猜（第 6、12 章）
- [ ] 是否需要分布式？——只在单机内存/磁盘真正不够时才考虑 Spark 等方案
- [ ] 现有管道是否已有明显反模式？——逐行循环、CSV 主存储、无谓的中间物化
- [ ] 是否在容器/cgroup 中运行？——确认 `POLARS_MAX_THREADS` 与实际配额一致（第 7 章）
- [ ] 基准数字是否注明了环境？——硬件、Polars 版本、数据规模缺一不可

## 练习

1. **环境确认**：安装 Polars 后打印 `pl.__version__` 和 `pl.thread_pool_size()`，并分别在 1、4、8 线程下运行本章的 1000 万行聚合基准（提示：`POLARS_MAX_THREADS` 需在 import 前设置，每次改完要重启进程）。记录耗时随线程数的变化。
2. **对比实验**：把 1.3 节基准中的 `pl.col("x").sum()` 分别换成 `.cast(pl.String)` 与 `.str.len_chars()`（先构造字符串列），观察数量级差异并解释原因。
3. **思考题**：你的团队当前有一个基于 pandas 的 50GB CSV 分析管道，迁移到 Polars 的前三个改造点是什么？（提示：从本章的反模式清单出发）
