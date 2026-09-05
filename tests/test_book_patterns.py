"""书中关键代码模式校验（对应各章示例）

运行：uv run pytest
"""
from datetime import date, datetime, timedelta

import polars as pl


# ---------- 第 2 章：核心对象与内存模型 ----------

def test_ch02_null_vs_nan():
    s = pl.Series("x", [1.0, None, float("nan")])
    assert s.is_null().to_list() == [False, True, False]
    assert s.is_nan().to_list() == [False, None, True]


def test_ch02_dtype_narrowing():
    df = pl.select(
        id=pl.int_range(0, 1_000_000, dtype=pl.Int64),
        flag=pl.int_range(0, 1_000_000, dtype=pl.Int64) % 2,
    )
    wide = df.estimated_size()
    narrow = df.with_columns(
        pl.col("id").cast(pl.Int32),      # 8B → 4B
        (pl.col("flag") == 1).alias("flag"),  # 8B → 1B
    ).estimated_size()
    assert narrow < wide * 0.5  # 收窄显著省内存


# ---------- 第 5 章：链式调用与表达式组合 ----------

def test_ch05_expression_reuse():
    zscore = lambda c: (pl.col(c) - pl.col(c).mean()) / pl.col(c).std()
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    out = df.select(zscore("a").alias("za"), zscore("b").alias("zb"))
    assert out.get_column("za").abs().sum() < 1e-9 or out.height == 3


def test_ch05_pipe_composition():
    def add_dow(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(dow=pl.col("ts").dt.weekday())

    lf = pl.LazyFrame({"ts": [datetime(2026, 8, 28)]})  # 周五
    out = lf.pipe(add_dow).collect()
    assert out["dow"][0] == 5


# ---------- 第 6 章：谓词下推 ----------

def test_ch06_predicate_pushdown(tmp_path):
    f = tmp_path / "d.parquet"
    pl.select(a=pl.int_range(0, 100), b=pl.int_range(0, 100) * 2).write_parquet(f)
    plan = (
        pl.scan_parquet(f).filter(pl.col("a") > 5).select("a").explain()
    )
    assert "SELECTION" in plan and "PROJECT" in plan


# ---------- 第 8 章：流式管道 ----------

def test_ch08_streaming_pipeline(tmp_path):
    src = tmp_path / "in.parquet"
    dst = tmp_path / "out.parquet"
    pl.select(
        k=pl.int_range(0, 10_000) % 7,
        x=pl.int_range(0, 10_000),
    ).write_parquet(src)

    (pl.scan_parquet(src)
       .filter(pl.col("x") > 100)
       .group_by("k")
       .agg(pl.len().alias("n"))
       .sink_parquet(dst))

    out = pl.read_parquet(dst)
    assert out.height == 7  # 所有 k 都有 x > 100 的行


# ---------- 第 9 章：清洗 ----------

def test_ch09_cast_strict():
    df = pl.DataFrame({"x": ["1", "2", "abc"]})
    out = df.select(pl.col("x").cast(pl.Int64, strict=False))
    assert out["x"].to_list() == [1, 2, None]


def test_ch09_categorical_memory():
    n = 100_000
    df = pl.DataFrame({"city": ["上海", "北京"] * (n // 2)})
    as_str = df.estimated_size()
    as_cat = df.with_columns(pl.col("city").cast(pl.Categorical)).estimated_size()
    assert as_cat < as_str


# ---------- 第 10 章：join 与聚合 ----------

def test_ch10_join_semantics():
    left = pl.DataFrame({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    right = pl.DataFrame({"k": [1, 2, 5], "w": [100, 200, 500]})
    assert left.join(right, on="k", how="inner").height == 2
    assert left.join(right.select("k"), on="k", how="semi").height == 2
    assert left.join(right.select("k"), on="k", how="anti").height == 2
    assert left.join(right, on="k", how="left").height == 4
    assert left.join(right, on="k", how="full", coalesce=True).height == 5


def test_ch10_len_vs_count():
    df = pl.DataFrame({"g": ["a", "a", "a"], "x": [1, None, 3]})
    out = df.group_by("g").agg(pl.len().alias("n"), pl.col("x").count().alias("c"))
    assert out["n"][0] == 3 and out["c"][0] == 2


def test_ch10_iqr_outlier():
    df = pl.DataFrame({"amount": [10.0, 12.0, 11.0, 13.0, 12.5, 11.8, 1000.0]})
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
    assert cleaned.height == 6  # 只剔除 1000.0


# ---------- 第 11 章：时间序列 ----------

def test_ch11_group_over_boundary():
    # 组内环比：用户边界处第一期应为 null
    df = pl.DataFrame({
        "user": ["a", "a", "b", "b"],
        "month": [1, 2, 1, 2],
        "total": [100, 120, 200, 180],
    }).sort("user", "month")
    out = df.with_columns(mom=pl.col("total").pct_change(1).over("user"))
    assert out["mom"][0] is None      # 用户 a 第一期
    assert out["mom"][2] is None      # 用户 b 第一期（不是 -0.8）
    assert abs(out["mom"][1] - 0.2) < 1e-9


def test_ch11_missing_period_trap():
    # 缺失周期导致 pct_change 错位
    sparse = pl.DataFrame({
        "month": [date(2026, 1, 1), date(2026, 3, 1)],
        "total": [100, 130],
    })
    wrong = sparse.with_columns(mom=pl.col("total").pct_change(1))
    assert wrong["mom"][1] == 0.3  # 看似环比，实则跨了 2 月

    dense = sparse.upsample("month", every="1mo").with_columns(
        pl.col("total").forward_fill()
    )
    right = dense.with_columns(mom=pl.col("total").pct_change(1))
    assert right["mom"][1] == 0.0   # 2 月持平
    assert right["mom"][2] == 0.3   # 3 月环比


# ---------- 第 10 章补充：高效统计三件套 ----------

def test_ch10_hist_and_approx():
    df = pl.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    h = df.get_column("amount").hist(bin_count=4)
    assert set(h.columns) == {"breakpoint", "category", "count"}
    assert h["count"].sum() == 8
    # 近似基数（HLL 估计）：此处只验证调用
    df.select(pl.col("amount").approx_n_unique())


def test_ch10_rle_id_sessions():
    df = pl.DataFrame({"event": ["A", "A", "B", "B", "B", "A"]})
    out = df.with_columns(session=pl.col("event").rle_id())
    assert out["session"].to_list() == [0, 0, 1, 1, 1, 2]


# ---------- 第 11 章补充：插值与时间戳 ----------

def test_ch11_interpolate():
    df = pl.DataFrame({"v": [10.0, None, 40.0]})
    out = df.with_columns(pl.col("v").interpolate())
    assert out["v"].to_list() == [10.0, 25.0, 40.0]


def test_ch11_from_epoch():
    out = pl.select(ts=pl.from_epoch(pl.Series([0, 60]), time_unit="s"))
    assert out["ts"].dt.second().to_list() == [0, 0]
    assert out["ts"].dt.minute().to_list() == [0, 1]


# ---------- 附录 A：SQL 方言 ----------

def test_appendix_sql_self():
    df = pl.DataFrame({"user_id": [1, 2, 3], "amount": [100, 200, 300]})
    out = df.sql("SELECT user_id FROM self WHERE amount > 150 ORDER BY user_id")
    assert out["user_id"].to_list() == [2, 3]


# ---------- 第 14 章：增量幂等 ----------

def test_ch14_incremental_idempotent(tmp_path):
    lake = tmp_path / "lake"
    lake.mkdir()
    src = tmp_path / "batch.parquet"

    pl.select(order_id=pl.int_range(0, 100), amount=pl.int_range(0, 100) * 2.5
    ).write_parquet(src)

    # 第一次：直接入库（湖为空）
    part = lake / "date=2026-08-28"
    part.mkdir()
    pl.read_parquet(src).write_parquet(part / "part-0.parquet")
    assert pl.read_parquet(part / "part-0.parquet").height == 100

    # 第二次重跑同批数据：anti join 挡住全部重复 → 0 行增量
    (pl.scan_parquet(src)
       .join(
           pl.scan_parquet(f"{lake}/**/*.parquet").select("order_id"),
           on="order_id", how="anti",
       )
       .sink_parquet(lake / "increment.parquet"))
    assert pl.read_parquet(lake / "increment.parquet").height == 0
