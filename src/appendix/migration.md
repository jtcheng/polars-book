# 附录 C 版本迁移与 deprecated 追踪

> Polars 迭代快，本附录记录常见迁移点与追踪方法。

## C.1 追踪方法

- `DeprecationWarning` 全量打印：`python -W error::DeprecationWarning`
- 官方 release notes：每版本的 API 变更清单

## C.2 常见迁移点

| 旧 | 新 |
|---|---|
| `lazy()` 显式转换 | `scan_*` 直接返回 LazyFrame |
| `apply` | `map_elements`（语义区分更明确） |
| `pl.count()` | `pl.len()` |
| `with_column` | `with_columns` |
| `groupby` | `group_by` |
| 旧流式 `collect(streaming=True)` | `collect(engine="streaming")` |

## C.3 锁版本建议

- 生产环境锁定 minor 版本 + 关注 release notes
- CI 中跑升级冒烟测试
