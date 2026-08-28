"""示例 1：基础与内存模型（对应第 1、2 章）

运行：uv run python examples/01_basics.py
"""
import time

import numpy as np
import polars as pl

print(f"polars {pl.__version__}, 线程池: {pl.thread_pool_size()}")

# --- 第 2 章：三个核心对象 ---
df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["a", "b", "c"],
    "score": [90.5, 85.0, 78.5],
})
print("schema:", dict(df.collect_schema()))

# Expr 是计算描述，不是执行
expr = pl.col("score") * 2 + 1
print("expr 类型:", type(expr).__name__)

# --- 第 2 章：dtype 收窄与内存 ---
big = pl.select(
    id=pl.int_range(0, 1_000_000, dtype=pl.Int64),
    flag=pl.int_range(0, 1_000_000, dtype=pl.Int64) % 2,
)
print(f"Int64 存储: {big.estimated_size('mb'):.2f} MB")
narrow = big.with_columns(
    pl.col("id").cast(pl.Int32),
    (pl.col("flag") == 1).alias("flag"),
)
print(f"收窄后:  {narrow.estimated_size('mb'):.2f} MB")

# --- 第 2 章：null 与 NaN ---
s = pl.Series("x", [1.0, None, float("nan")])
print("is_null:", s.is_null().to_list())   # [False, True, False]
print("is_nan: ", s.is_nan().to_list())    # [False, False, True]

# --- 第 2 章：零拷贝 ---
arr = np.arange(1_000_000, dtype=np.float64)
ser = pl.Series("x", arr)
same = ser.to_numpy().__array_interface__["data"][0] == arr.__array_interface__["data"][0]
print("零拷贝:", same)

# --- 第 1 章：1 亿行聚合基准 ---
N = 100_000_000
t0 = time.perf_counter()
result = (
    pl.select(k=pl.int_range(0, N) % 1_000, x=pl.int_range(0, N) % 1_000_000)
      .group_by("k")
      .agg(pl.col("x").sum())
)
elapsed = time.perf_counter() - t0
print(f"1 亿行 group_by 聚合: {elapsed:.2f}s, 输出 {result.height} 组")
