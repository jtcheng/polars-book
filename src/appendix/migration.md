# 附录 C 版本迁移与 deprecated 追踪

> Polars 迭代快，本附录记录常见迁移点与追踪方法。

## C.1 追踪方法

- `DeprecationWarning` 全量打印：`python -W error::DeprecationWarning`
- 官方 release notes 与升级指南（docs.pola.rs/releases/upgrade）：每版本的 API 变更清单
- 大版本升级后全量检索两个**类型化移除异常**：`AttributeRemovedError`（方法/属性被移除）与 `ArgumentRemovedError`（参数被移除改名）——2.0 起被移除的 API 抛的都是这两类异常，错误信息直接指向替代写法（如 `melt` → `unpivot`、`join_nulls` → `nulls_equal`），CI 里 catch 它们即可精确定位待迁移代码

两条路径覆盖不同的时间尺度：warning 是"当下"——升级后把弃用当场变成异常抛出，在 API 真正移除（通常发生在弃用后若干版本）之前留出迁移窗口；release notes 是"将来"——逐版本的 API 变更清单，用来预判哪些变更会落到自己头上。只靠其中一条都会漏：warning 看不到尚未弃用但即将变更的 API，notes 则不会指出你的代码哪一行踩中了变更——前者要主动跑全量测试，后者要人工对照清单。两者配合，锁版本下的例行升级才不会变成赌运气。

## C.2 版本路线

| 版本 | 关键节点 |
|---|---|
| 0.19–0.20 前后 | API 大改名期：`groupby` → `group_by`、`with_column` → `with_columns`、`apply` → `map_elements`、`melt` → `unpivot`（1.0 前叫 `melt`）、`dtypes` → `schema_overrides` 等 |
| 1.0（2024-07） | 稳定 API 承诺：1.x 系列内不再破坏式变更，生产可以放心锁 1.x |
| 1.25 | `collect(streaming=True)` 参数改 `engine`；`explain(streaming=True)` 弃用 |
| 1.41 | 新流式引擎进入 stable，取代旧流式引擎（第 8 章） |
| 1.42 | `concat(how="horizontal")` 语义收紧：默认行为弃用，严格版要求行数相等 |
| 1.43 | `profile()` 弃用（旧引擎专用，新流式引擎下耗时统计不可靠） |
| 1.44 | 前一版实测基线 |
| 2.0.0 | 本书写作与实测基线。**核心变更：流式引擎成为 `collect()` 默认**（`engine="auto"` 解析为流式；join/group_by/unpivot 不再保证行序，见第 8 章）；核外溢写默认启用（约 80% RAM 起溢写、磁盘预算 64GB）。API 层面：`profile()` 移除；rolling 系列参数 `min_periods` 改名 `min_samples`；`Categorical` 构造改为 `categories=`（新增命名字典 `pl.Categories`）；`get_categories` 移除；`replace` 的 `default`/`return_dtype` 移除（改用 `replace_strict`）；字符串 cast 到时间类型移除（改用 `.str.to_datetime()` 等专用解析）；整数 ↔ Categorical/Enum 互 cast 禁止（改用 `.cat.to()` / `.cat.physical()`）；IO 函数的 `rechunk`、`Series.to_numpy(use_pyarrow=)`、各优化标志散参数（改收进 `optimizations=pl.QueryOptFlags(...)`）等移除；`cut`/`qcut` 开始弃用（改用 `bin_intervals`/`bin_quantiles`） |

## C.3 常见迁移点

| 旧写法 | 新写法 |
|---|---|
| `df.lazy()` 显式转换 | `scan_*` 直接返回 LazyFrame（无需 `lazy()`） |
| `apply` | `map_elements`（语义区分更明确；0.20 期间已移除） |
| `pl.count()` | `pl.len()`（len 计行、count 计非 null，语义统一） |
| `with_column` | `with_columns`（1.0 前已移除） |
| `groupby` | `group_by`（1.0 前已移除） |
| `df.melt(id_vars, value_vars)` | `df.unpivot(index, on)` |
| `read_csv(dtypes=...)` / `scan_csv(dtypes=...)` | `schema_overrides=...` |
| `collect(streaming=True)` | `collect(engine="streaming")`（1.25 起改参数） |
| `join(how="outer")` | `join(how="full")`（`outer` 0.20.29 起弃用） |
| `concat(how="horizontal")` | `how="horizontal", strict=True`（行数须相等）或 `how="horizontal_extend"`（不等补 null） |
| `pl.enable_string_cache()` | 不再需要——全局 Categorical 跨字典 join 自动 remap；2.0 起共享字典用命名字典 `pl.Categorical(categories="name")` 表达（混用名字典直接报错，快失败） |
| `rolling_*(min_periods=)` | `min_samples=`（2.0 改名） |
| `s.cat.get_categories()` | `s.unique()`（查看取值；Enum 的类别表用 `dtype.categories`，2.0 已移除该方法） |
| `lf.profile()` 定位热点 | 1.43 起弃用、2.0 已移除——改用基准计时 + 实测峰值 RSS（第 8、12 章） |
| `Categorical(ordering=)` | 2.0 移除——排序恒为 lexical，按出现顺序排用 `to_physical()`；构造改用 `Categorical(categories=...)` |
| `replace(..., default=)` / `return_dtype=` | 2.0 移除——设默认值改用 `replace_strict` |
| `col.cast(pl.Date)`（字符串列） | 2.0 移除——改用 `.str.to_date()` / `.str.to_datetime()` / `.str.to_time()`（可指定格式，控制更精细） |
| `s.cast(pl.Categorical)` / `s.cast(pl.Enum(...))`（整数列） | 2.0 禁止——改用 `.cat.to(dtype)`；反向取物理索引用 `.cat.physical()` |
| `pl.Categorical("physical")` | ⚠️ 2.0 陷阱：不再报错，而是**静默创建名为 "physical" 的命名字典**——旧代码以为在选排序语义，实际拿到的是独立编码空间，与其他列 join 会因字典名不同而报错。升级前全文检索这一写法 |
| `with_row_count()` | `with_row_index()`（默认列名由 `row_nr` 改为 `index`） |
| `cut(breaks)` / `qcut(...)` | 2.0 起弃用——改用 `bin_intervals` / `bin_quantiles`（注意默认**左闭**，原右闭语义需传 `right_closed=True`；`labels` 必填） |
| `collect(predicate_pushdown=...)` 等散优化标志 | 2.0 收进 `optimizations=pl.QueryOptFlags(predicate_pushdown=...)` |
| `pl.threadpool_size()` | `pl.thread_pool_size()` |

## C.4 锁版本建议

- 生产环境锁定 minor 版本 + 关注 release notes
- CI 中跑升级冒烟测试（搭配 `python -W error::DeprecationWarning` 提前暴露弃用）
