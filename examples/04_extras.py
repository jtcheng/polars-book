"""示例 4：官方文档核查后补充的高频 API（对应第 7、8、10、11 章、附录 A）

运行：uv run python examples/04_extras.py
"""
from pathlib import Path

import polars as pl

WORK = Path("example_data")
WORK.mkdir(exist_ok=True)

# --- 第 10 章：approx_n_unique——大数据集的基数近似 ---
big = pl.select(
    user_id=pl.int_range(0, 1_000_000) % 997,   # 真实基数 997
    amount=(pl.int_range(0, 1_000_000) % 10_000).cast(pl.Float64),
)
exact = big.select(pl.col("user_id").n_unique())["user_id"][0]
approx = big.select(pl.col("user_id").approx_n_unique())["user_id"][0]
print(f"基数：精确 {exact} vs 近似 {approx}")   # 几乎一致，但近似版远快

# --- 第 10 章：hist——一行看分布 ---
h = big.get_column("amount").hist(bin_count=5)
print("\namount 分布直方图:")
print(h)

# --- 第 10 章：rle_id——识别连续相同段 ---
events = pl.DataFrame({
    "user": [1, 1, 1, 1, 2, 2],
    "state": ["A", "A", "B", "B", "A", "A"],
})
print("\n连续状态段编号（会话切分的基础）:")
print(events.with_columns(
    seg=pl.col("state").rle_id().over("user"),
))

# --- 第 11 章：from_epoch + interpolate ---
logs = pl.select(ts=pl.from_epoch(pl.Series([1756000000, 1756000060, 1756000120]),
                                  time_unit="s"))
print("\nUnix 时间戳转 Datetime:")
print(logs)

ts = pl.DataFrame({"v": [10.0, None, None, 40.0]})
print("\n缺失值线性插值（vs forward_fill）:")
print(ts.with_columns(
    interp=pl.col("v").interpolate(),
    ffill=pl.col("v").forward_fill(),
))

# --- 第 8 章：lazy sink——非阻塞式流式写出 ---
src = WORK / "src.parquet"
big.head(100_000).write_parquet(src)

sink_lf = (
    pl.scan_parquet(src)
      .group_by("user_id")
      .agg(pl.col("amount").mean().alias("avg_amount"))
      .sink_parquet(WORK / "sink_lazy.parquet", lazy=True)  # 只登记不执行
)
# ...此处可继续登记其它 sink，统一调度...
sink_lf.collect()      # 此刻才真正流式执行，返回空 DataFrame（数据已落盘）
result = pl.read_parquet(WORK / "sink_lazy.parquet")   # 从落盘文件读取结果
print(f"\nlazy sink 产出: {result.height} 组")

# --- 附录 A：DataFrame.sql——self 表引用 ---
print("\ndf.sql() 的 self 引用:")
print(result.sql("SELECT * FROM self WHERE avg_amount > 400 ORDER BY user_id LIMIT 3"))

# 清理
import shutil
shutil.rmtree(WORK)
print("\n示例数据已清理")
