# 第 7 章 并行与多线程模型

> 本章要解决什么问题：理解 Polars 如何利用多核——Rayon 调度、morsel-driven 并行，以及为什么 GIL 在这里不是瓶颈。

## 7.1 线程池与 Rayon 工作窃取

```mermaid
sequenceDiagram
    participant M as 主线程
    participant T1 as 工作线程 1
    participant T2 as 工作线程 2
    participant T3 as 工作线程 3
    M->>M: 接收列式数据
    M->>T1: 分发 morsel 块
    M->>T2: 分发 morsel 块
    M->>T3: 分发 morsel 块
    T1-->>M: 结果块
    T2-->>M: 结果块
    T3->>T1: 队列空 → 窃取 T1 未完成块
    T3-->>M: 窃取块结果
    M->>M: 归并输出
```

- Rayon：work-stealing 调度器，自动负载均衡
- `pl.thread_pool_size()` 查看与控制

```python
import polars as pl
print(pl.thread_pool_size())   # 例如 10 —— 默认等于物理核数

# 感受并行度带来的收益：1 亿行聚合
import time

N = 100_000_000
df = pl.DataFrame({"x": pl.int_range(0, N, dtype=pl.Float64)})

t0 = time.perf_counter()
df.select(pl.col("x").sum())
print(f"并行 sum: {time.perf_counter() - t0:.3f}s")
# 改变线程数后对比（需重启进程）：
# POLARS_MAX_THREADS=1 python bench.py   # 单线程
# POLARS_MAX_THREADS=4 python bench.py   # 4 线程
```

## 7.2 morsel-driven 并行

- 数据切成小块（morsel）而非按行分发
- 与操作系统页缓存对齐

```python
# morsel 粒度由引擎自动决定，但能从行为上观察它
# filter 这类逐行无状态操作天然可并行——每个 morsel 独立处理
big = pl.scan_parquet("big.parquet")
(big.filter(pl.col("a") > 0)
    .select(pl.col("a").sum())
    .collect())
# 扫描、过滤、求和全部按 morsel 并行，无全局屏障
```

## 7.3 GIL 为何不是瓶颈

- 执行在 Rust 侧，释放 GIL
- 只在 collect 边界回到 Python

```python
# GIL 边界示意：Python 负责描述，Rust 负责执行
result = (
    pl.scan_parquet("big.parquet")   # Python：只是记账
      .group_by("k")                # Python：还是记账
      .agg(pl.len())                # Python：依然是记账
      .collect()                     # ← 唯一的边界：进 Rust，释放 GIL，多线程执行
)
```

这带来一个重要推论：**在 Polars 执行期间，你可以同时运行其他 Python 任务**（如 I/O），两者互不阻塞。

## 7.4 线程数控制

- 环境变量 `POLARS_MAX_THREADS`
- 与其他进程竞争 CPU 时的隔离策略

```bash
# 必须在 import polars 之前设置
export POLARS_MAX_THREADS=8
python pipeline.py
```

```python
# 典型场景：容器配额 4 核，但机器 64 核
# 不设置 → Polars 默认用 64 线程 → 严重过订阅 → 上下文切换淹没收益
# 对策：cgroup 感知地显式设置
import os
os.environ["POLARS_MAX_THREADS"] = "4"  # 必须在 import polars 前
import polars as pl
print(pl.thread_pool_size())  # 4
```

## 7.5 过度并行的陷阱

```python
# ❌ 反模式：进程池 × Polars 线程池 = 线程爆炸
from concurrent.futures import ProcessPoolExecutor

def process(path: str):
    return pl.scan_parquet(path).group_by("k").agg(pl.len()).collect()

# 8 进程 × 每进程 10 线程 = 80 线程争抢 10 核
with ProcessPoolExecutor(8) as ex:
    list(ex.map(process, paths))

# ✅ 正确做法：要么单进程让 Polars 自己并行（推荐）
# 要么手动限制每进程线程数：POLARS_MAX_THREADS=2 + 5 进程
```

## 7.5 GPU 引擎（可选）

Polars 支持将查询下推到 NVIDIA GPU 执行（`engine="gpu"`，需额外安装 `polars[gpu]` 且依赖 CUDA 环境）：

```python
# 安装：uv add "polars[gpu]"
result = (
    pl.scan_parquet("big.parquet")
      .group_by("k").agg(pl.col("x").sum())
      .collect(engine="gpu")    # 数据自动搬运到显存执行
)
```

适用边界：大规模数据 + 高算术密度的查询获益最大；小数据反而因主机↔显存搬运亏本。CPU 流式引擎（第 8 章）依然是内存受限场景的首选。

## 要点回顾

- Polars 并行无需手动配置，但可干预
- morsel 粒度是流式引擎（第 8 章）的执行单元
- 进程级并行与 Polars 线程池叠加会造成过订阅

## 性能检查清单

- [ ] 是否在容器/cgroup 中正确设置了线程数？
- [ ] 是否因嵌套并行（自建 multiprocessing + Polars）造成过订阅？
- [ ] `POLARS_MAX_THREADS` 是否在 import polars 之前设置？
- [ ] 是否利用了 GIL 释放做 I/O 与计算重叠？

## 练习

1. **扩展实验**：把 7.1 节的基准改为 `rolling_mean(100)` 与 `sort()`，观察哪种操作从多线程获益更多，并解释（提示：行间依赖）。
2. **过订阅复现**：用 `ProcessPoolExecutor(4)` 配合默认线程池跑四个聚合任务，再用 `POLARS_MAX_THREADS=2` 重跑，对比总耗时与系统负载。
3. **思考题**：Polars 执行期间用 `threading` 启动一个下载任务，为什么不会与聚合争抢 GIL？
