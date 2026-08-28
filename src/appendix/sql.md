# 附录 A Polars SQL 方言

> 大部分 Polars 操作可用 SQL 表达，适合熟悉 SQL 的读者快速上手。

## A.1 SQL 上下文

```python
result = pl.sql("""
    SELECT user_id, SUM(amount) AS total
    FROM df
    WHERE date >= '2026-01-01'
    GROUP BY user_id
    ORDER BY total DESC
""").collect()
```

除了模块级 `pl.sql()`，还可以在 DataFrame/LazyFrame 上直接执行 SQL——`self` 即当前表：

```python
df.sql("SELECT user_id, amount FROM self WHERE amount > 100")
lf.sql("SELECT city, COUNT(*) AS n FROM self GROUP BY city")  # 返回 LazyFrame
```

## A.2 与标准 SQL 的差异

以下每条差异都在 polars 1.44 实测验证（1.44 尚无 `DATE_TRUNC`/`QUANTILE` 的 SQL 函数，写作时以运行时报错为准）：

- **NULL 判断写 `IS NULL`，没有 `is_null()` 函数**：SQL 侧与标准 SQL 一致用 `IS NULL` / `IS NOT NULL`；表达式 API 侧的 `is_null()` 不能搬进 SQL。`WHERE x = NULL` 与标准 SQL 行为一致——不匹配任何行：

  ```python
  pl.sql("SELECT * FROM df WHERE amount IS NULL").collect()
  ```

- **`LIMIT`/`OFFSET` 可用，`TOP` 不可用**：支持 `LIMIT n OFFSET m`；T-SQL 风格的 `SELECT TOP n` 会报错（提示改用 `LIMIT`）。无 `ORDER BY` 时不要依赖返回行序——先排序再分页：

  ```python
  pl.sql("SELECT user_id FROM df ORDER BY amount DESC LIMIT 10 OFFSET 20").collect()
  ```

- **函数覆盖不完整**：`MEDIAN`、`COALESCE`、`NULLIF`、`STRFTIME`、`EXTRACT` 都有；但 `DATE_TRUNC`、`QUANTILE` 没有对应——周期截断改用 `STRFTIME` 提取，分位数只能回表达式 API（`pl.col("x").quantile(0.9)`）：

  ```python
  pl.sql("SELECT STRFTIME(ts, '%Y-%m') AS month, SUM(amount) FROM df GROUP BY month").collect()
  ```

- **字符串函数细节不同**：`LIKE` 大小写敏感（`'a%'` 匹配不到 `'Alice'`，忽略大小写用 `ILIKE`）；拼接 `||` 和 `+` 都支持；表达式侧的 `str.contains` 在 SQL 里没有同名函数：

  ```python
  pl.sql("SELECT * FROM df WHERE note ILIKE 'a%'").collect()
  ```

## A.3 适用场景与局限

交互式探索阶段，SQL 往往更快上手：一行 `pl.sql()` 直接出结果，不必翻表达式 API 文档；团队里已有 SQL 技能储备时，用它做临时验证、口径核对都很顺手。管道产出 Parquet 之后的 ad-hoc 查询也是 SQL 的舒适区——第 13、14 章的 DuckDB 协作模式就是"Polars 做重变换、SQL 做探索"的分工。

生产管道则推荐表达式 API。表达式是 Python 对象，可存变量、函数化封装、`pipe` 拼进链式管道（第 5 章）；列名与参数能被 IDE 补全和静态检查捕获，而 SQL 是字符串——列名拼错要等到运行时才报错。此外 SQL 侧覆盖不了全部表达式能力（如 `over` 窗口、`join_asof`、`group_by_dynamic`），复杂逻辑硬塞进 SQL 反而更难维护。

两者并不互斥：`pl.sql()` 的返回值是 LazyFrame，可以继续 `.filter().sink_parquet()` 接回表达式管道。实践建议是——用它换上手速度，别用它换可维护性。
