# 第 2 章 核心对象与内存模型

> 本章要解决什么问题：掌握 DataFrame/Series/Expr 三个核心对象，理解 Arrow 列式内存布局如何成为 Polars 性能的地基，并建立 API 命名空间全景。

## 2.1 三个核心对象

```mermaid
classDiagram
    class DataFrame {
        +Vec~Series~ columns
        +height
        +select() Expr 上下文
        +filter() Expr 上下文
        +group_by() 分组上下文
    }
    class Series {
        +dtype
        +name
        +to_numpy()
        +to_arrow()
    }
    class Expr {
        +col/lit/when
        +str/dt/cat/list/struct
        +map_elements(func)
    }
    DataFrame "1" o-- "n" Series : 列组成
    Series <.. Expr : 表达式求值于
    DataFrame ..> Expr : select/with_columns 上下文
```

- DataFrame：列的集合，不是行的容器
- Series：带类型的一维列
- Expr：**尚未执行的计算描述**——Polars 性能哲学的起点

三者是层层递进的关系：DataFrame 持有若干 Series，每个 Series 由一段（或多段）带 dtype 的连续缓冲组成——引擎以 chunk 为单位管理，多文件 scan 与 `concat` 天然产生多 chunk，必要时用 `rechunk()` 合并（第 7 章）；而 Expr 只是描述"对哪些列做什么"的一棵计算树——它不持有数据，放进 select/filter 这类上下文之前不会发生任何求值。把"数据"与"计算描述"拆成两类对象，是后续一切优化的前提：计算意图以 Expr 的形式独立存在，引擎才能在执行前看到完整的计算树，做谓词下推与投影裁剪（第 4 章）。"列的集合"这一定位还有一层实际后果：整表 schema 在构造时就已确定，按列取数据是零成本的原生操作，而取一行要把各个列缓冲区的同一位置拼起来，天然走的是例外路径。

### 动手感受三个对象

```python
import polars as pl

# DataFrame：列的集合
df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["a", "b", "c"],
    "score": [90.5, 85.0, 78.5],
})
print(df.schema)
# Schema({'id': Int64, 'name': String, 'score': Float64})
# 注意：dtype 是强类型的，不存在 pandas 的 object 兜底

# Series：带类型的一维列
s = df.get_column("score")
print(s.dtype, s.len())      # Float64 3
print(s.sum(), s.mean())     # 254.0 84.666...

# Expr：尚未执行的计算描述
expr = pl.col("score") * 2 + 1
print(type(expr))            # <class 'polars.expr.expr.Expr'>
# 此时没有任何计算发生——它只是"描述"
```

关键心智模型：`pl.col("score") * 2 + 1` 在 pandas 里是**立即求值**的 NumPy 向量化运算——每个算子各扫一遍内存（本例两趟遍历、两个中间数组）；在 Polars 里它是一棵待编译的表达式树，求值时多个算子可**融合**为一个 Rust 向量化内核，一趟遍历完成（第 4 章）。差异不在"向量化与否"，而在"逐算子执行还是整树融合执行"。

## 2.2 Arrow 列式内存布局

```mermaid
flowchart LR
    subgraph row["Row-major（pandas/NumPy 对象模式）"]
        R1["行1: [id, name, age]"]
        R2["行2: [id, name, age]"]
        R3["行3: [id, name, age]"]
    end
    subgraph col["Column-major（Arrow/Polars）"]
        C1["id:   1|2|3|... 连续内存"]
        C2["name: a|b|c|... 连续内存"]
        C3["age: 30|25|... 连续内存"]
    end
    row -->|"缓存不友好<br/>分支预测失败"| CPU["CPU 缓存行"]
    col -->|"顺序预取<br/>SIMD 友好"| CPU
```

- 与 NumPy row-major 对比：缓存局部性
- validity bitmap：null 不是 NaN，也不是哨兵值
- 零拷贝原理：为什么 `from_arrow` / `to_numpy`(无 null) 不复制数据

列式布局的价值要用 CPU 的视角才能看清：同一列的数据在内存里连续存放，扫描时预取器能整段装进缓存；row-major 把同一行的多个字段交错存放，每一步只消费每个缓存行的一小部分，等于反复为用不上的字节买单。null 的表示是一块与数据并排的位图，每个值一个比特，标记"这一格有没有值"——因此 null 是独立于取值空间的标记，不占数据位，也不会与任何真实值（包括 NaN）混淆。零拷贝是前两者的直接推论：Arrow 与无 null 的 NumPy 数组的物理布局与 Polars 一致，交接只需传递缓冲区指针；反之，凡对方没有能力表达的成分（如 NumPy 没有位图层），就必须复制重建。

> **精确表述**：Polars 遵循 Arrow 列式规范，但**并非构建在 PyArrow 之上**——它有自己的 Rust 计算与缓冲区实现（衍生自 Arrow2 的设计）。这带来两个推论：① 与 Arrow 生态（PyArrow、DuckDB、DataFusion）交换数据时通常零拷贝；② Polars 的内部优化不受 PyArrow 限制，validity bitmap 的处理效率比通用 Arrow 实现更高。

### 观察内存占用

```python
import polars as pl

df = pl.select(
    id=pl.int_range(0, 1_000_000, dtype=pl.Int64),
    flag=(pl.int_range(0, 1_000_000, dtype=pl.Int64) % 2 == 0),
)

print(df.estimated_size("mb"))  # ≈ 7.7 MB
# Int64 (8B) + Bool (1B) × 100 万行
# Bool 按位打包：每行约 8 + 1/8 = 8.125 字节

# 收窄 dtype 立省一半
df2 = df.with_columns(pl.col("id").cast(pl.Int32))
print(df2.estimated_size("mb"))  # ≈ 3.9 MB
```

### null 与 NaN 是两回事

```python
s = pl.Series("x", [1.0, None, float("nan")])
print(s.is_null())   # false / true / false —— 值缺失
print(s.is_nan())    # false / null / true —— 浮点未定义；null 位置传播为 null
print(s.sum())       # nan —— nan 会传染，null 会被跳过（1.0 + nan = nan）
# 聚合前先处理：
print(s.fill_nan(0).sum())   # 1.0 —— 先 fill_nan 再聚合
```

### 零拷贝验证

```python
import numpy as np

arr = np.arange(1_000_000, dtype=np.float64)
s = pl.Series("x", arr)          # 无 null → 零拷贝
print(s.to_numpy().__array_interface__["data"][0]
      == arr.__array_interface__["data"][0])  # True：同一块内存
```

## 2.3 dtype 系统

- 数值类型严格固定宽度（`Int8`~`Int64`、`Float32/64`）
- `String` 不是 Python str 对象的集合
- `Categorical/Enum`：字典编码（第 9 章展开）

这三条背后是同一个设计决定：dtype 不只是标签，它直接规定数据在内存里的物理形态，进而决定哪些操作能在连续内存上向量化完成——固定宽度让寻址变成偏移量算术，String 用独立缓冲区避免 PyObject 指针数组，字典编码把字符串比较换成整数比较。选错 dtype 的代价因此不是"风格不好"，而是整列操作退回慢路径。下面的全景表按五族给出可选清单，逐族的取舍与坑在后续小节展开。

### dtype 全景

| 分组 | dtype | 一句话定位 |
|---|---|---|
| 整数 | `Int8/16/32/64`、`UInt8/16/32/64` | 严格固定宽度，溢出即报错，没有 Python int 的无限弹性 |
| 浮点 | `Float32/Float64` | IEEE 754；null 与 NaN 是两套语义（见 2.2） |
| 时间 | `Date`、`Datetime`、`Duration` | `Date` 是 i32 天数；`Datetime` 构造默认微秒精度 |
| 嵌套 | `List`、`Array`、`Struct` | 变长列表 / 定宽定长 / 一行内的命名字段 |
| 编码 | `String`、`Categorical`、`Enum` | 原生字符串缓冲 / 动态字典 / 固定字典 |

时间类型的默认精度值得亲手确认一次：

```python
import datetime as dt

df = pl.DataFrame({"ts": [dt.datetime(2026, 8, 1, 12, 30)]})
print(df.schema["ts"])
# Datetime(time_unit='us', time_zone=None)——Python datetime 构造默认落到微秒精度
# 需要纳秒：cast(pl.Datetime("ns"))；需要挂时区：`dt.replace_time_zone("Asia/Shanghai")`（naive→aware 走"重新解释墙钟时间"语义；已有时区做换算用 `dt.convert_time_zone`，两者勿混）
```

嵌套三兄弟一句话区分：`List` 每行长度可变（`[[1, 2], [3]]`）；`Array` 宽度写死在 dtype 里（`Array(Int64, 2)` 只装恰好两个元素，还能嵌套成多维）；`Struct` 是一行内的命名字段（`Struct({'x': Int64, 'y': String})`），相当于把宽表的一小段折叠进单列。

编码三选一的取舍：高基数或近似唯一 → `String`；低基数且类别未知、会增长 → `Categorical`（字典随数据动态生长）；低基数且类别固定已知 → `Enum`（编译期校验取值，排序语义随声明固定，第 9 章）。

### String 不是 Python str 对象的集合

pandas 的 object 列本质是 **PyObject 指针数组**：每个元素指向一个散落在 Python 堆上的 `str` 对象，任何比较/哈希/排序都要先解引用、再进解释器。Polars 的 `String` 列基于**变长字符串视图（binview）布局**：字符串字节集中存放在共享缓冲区，每行持有一个 16 字节定宽 view 结构（长度 + 缓冲区指针/偏移，外加 12 字节内联空间）——不超过 12 字节的字符串整个内联在 view 里，无需第二次访存；比较与哈希先比内联前缀即可快速淘汰大量候选。sort/filter/contains 等操作直接在缓冲区上向量化执行，全程不经过 Python 解释器。量级差异一句话：百万行 String 列 `sort()` 实测约 13 ms，同样数据转 Python 列表再 `sorted()` 约 104 ms——差的就是每个元素一次解释器对象开销（polars 1.44 / macOS arm64 实测）。

```python
import random

random.seed(0)
words = [f"city_{i}" for i in range(1000)]
col = [random.choice(words) for _ in range(1_000_000)]
s = pl.Series("w", col)

s.sort()      # ≈ 13 ms：直接在 Rust 缓冲区上向量化
sorted(col)   # ≈ 104 ms：百万次 PyObject 解引用 + 解释器比较
```

### dtype 检查工具

`df.schema` 返回"列名 → dtype"的有序映射，`df.dtypes` 只列类型序列。lazy 侧的正确姿势是 `lf.collect_schema()` **方法**——`LazyFrame` 的 `.schema` 属性在 1.x 已软废弃：属性访问仍能取到值，但每次都触发 `PerformanceWarning`（解析 lazy schema 需要推演整个查询计划，代价可能不小）。肌肉记忆：**eager 用属性、lazy 用方法**。

```python
lf = pl.LazyFrame({"a": [1, 2], "b": ["x", "y"]})
print(lf.collect_schema())   # Schema({'a': Int64, 'b': String})——官方姿势
# lf.schema                  # 1.44 实测仍可用，但触发 PerformanceWarning，别写进生产代码

df = pl.DataFrame({"a": [1, 2]})
print(df.schema)             # Schema({'a': Int64})（eager：schema 直接在手）
print(df.dtypes)             # [Int64]
```

```python
# dtype 决定行为：String 列的 sort 与 Categorical 的 sort 代价完全不同
df = pl.DataFrame({"city": ["上海", "北京", "上海", "深圳"]})
df_cat = df.with_columns(pl.col("city").cast(pl.Categorical))
print(df_cat.get_column("city").to_physical())  # 字典索引，如 [2, 3, 2, 4]
# 注意：物理编码的具体取值与分配顺序是实现细节，勿依赖其语义
```

## 要点回顾

- DataFrame 是列的集合；Expr 是计算的描述而非执行
- Arrow 列式布局带来缓存友好、SIMD 可用、跨系统零拷贝
- null（bitmap 标记）与 NaN（浮点值）语义不同，处理方式也不同
- 列式布局是后续流式引擎分块（morsel）处理的前提——morsel 是十万行量级的水平切片，列式布局保证每个 morsel 内每一列都是连续内存，可独立流过计算内核（第 7 章）

## 性能检查清单

- [ ] 数据是否以最窄的合理 dtype 存储？（用 `estimated_size()` 验证）
- [ ] 是否避免了把列转成 Python 对象列表再处理？（`to_list()` 是性能悬崖）
- [ ] null 与 NaN 是否被正确区分处理？
- [ ] 是否零拷贝衔接了 NumPy/Arrow 数据源？

## 练习

1. **dtype 收窄**：构造一个 `user_id`（范围 0~500 万）和 `flag`（0/1）的 DataFrame，用 `estimated_size()` 对比 Int64/Int32/Int8 三种存储的内存，并给出安全的最窄组合。
2. **null vs NaN**：对 Series `[1.0, None, nan]` 依次执行 `sum()`、`fill_null(0).sum()`、`fill_nan(0).sum()`，解释三个结果为什么不同。
3. **零拷贝验证**：用 `__array_interface__` 验证 `to_numpy()` 的零拷贝行为，然后给原 Series 加一个 null 再试一次——解释为什么这次不再零拷贝。
