# 附录 C 版本迁移与 deprecated 追踪

> Polars 迭代快，本附录记录常见迁移点与追踪方法。

## C.1 追踪方法

- `DeprecationWarning` 全量打印：`python -W error::DeprecationWarning`
- 官方 release notes：每版本的 API 变更清单

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
| 1.44 | 本书写作与实测基线 |

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
| `pl.enable_string_cache()` | 不再需要——跨字典 Categorical join 自动 remap（1.x 行为） |
| `s.cat.get_categories()` | `s.unique()`（查看取值；Enum 的类别表用 `dtype.categories`，2.0 将移除该方法） |
| `lf.profile()` 定位热点 | 1.43 起弃用——改用基准计时 + 实测峰值 RSS（第 8、12 章） |

## C.4 锁版本建议

- 生产环境锁定 minor 版本 + 关注 release notes
- CI 中跑升级冒烟测试（搭配 `python -W error::DeprecationWarning` 提前暴露弃用）
