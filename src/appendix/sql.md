# 附录 A Polars SQL 方言

> 大部分 Polars 操作可用 SQL 表达，适合熟悉 SQL 的读者快速上手。

## A.1 SQL 上下文

```python
result = pl.sql("""
    SELECT user_id, SUM(amount) AS total
    FROM df
    WHERE date >= '2024-01-01'
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

- 表引用：直接引用 Python 变量名
- 函数覆盖：表达式能力大多有对应 SQL 函数
- 与 `.sql()` 方法的关系

## A.3 适用边界

- 探索性查询：SQL 更快上手
- 生产管道：推荐表达式 API（编译期检查、优化器提示更友好）
