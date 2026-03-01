# Python 量化栈

> 对应学习计划 Phase 0：NumPy, Pandas, Matplotlib/Seaborn。

---

## 前言：为什么量化要用 Python 与这些库

量化研究处理的对象主要是**表格型数据**（股票代码、日期、开高低收、成交量、因子暴露、收益等）和**时间序列**（按时间排列的收益、价格、指标）。计算上大量涉及**向量与矩阵运算**（收益向量、协方差矩阵、因子矩阵）、**按组聚合**（按行业、按日期的统计）、**滚动计算**（移动平均、滚动波动率）以及**可视化**（收益曲线、分布图、热力图）。  
纯 Python 的列表与循环虽然能实现这些功能，但**效率低**、代码冗长且易错。**NumPy** 提供高效的多维数组与向量化运算，**Pandas** 在 NumPy 之上提供带标签的表格与时间序列，支持缺失值、合并、分组等，**Matplotlib** 与 **Seaborn** 负责绘图。三者构成了量化研究的**基础工具链**。  
本章按教材标准展开：先说明**环境与工具**，再系统讲解 **NumPy**（创建、索引、运算、广播）、**Pandas**（Series、DataFrame、读写、分组、时间序列）、**Matplotlib/Seaborn**（基本图与进阶图），并配有**可运行的代码示例、输出说明与常见错误**，最后给出**综合练习**以串联概率统计与量化栈。

---

# 第一章 环境与工具

**动机**：量化研究需要稳定、可复现的编程环境与数据处理、绘图能力；Python 配合 NumPy/Pandas/Matplotlib 等库可高效完成「数据读入 → 清洗 → 统计 → 可视化」全流程，本节明确环境与必装库。

## 1.1 Python 版本与发行版

- **Python 版本**：建议 **3.9 及以上**（3.10、3.11 均可）。量化常用库（NumPy、Pandas、scikit-learn、statsmodels 等）已普遍支持 3.9+；避免使用已停止维护的 2.x。  
- **发行版**：  
  - **Anaconda**：包含 Python、NumPy、Pandas、Matplotlib 等科学计算库，以及图形化环境管理；适合「开箱即用」。  
  - **Miniconda**：仅包含 conda 与 Python，需自行安装所需库；更轻量，适合已有使用习惯的用户。  
- **环境隔离**：建议为量化项目单独创建环境，避免与其它项目依赖冲突。  
  - 创建：`conda create -n quant python=3.10`  
  - 激活：`conda activate quant`（Windows 下在 Anaconda Prompt 或配置好的终端中执行）  
  - 安装包：`conda install numpy pandas matplotlib seaborn scipy jupyter` 或 `pip install numpy pandas matplotlib seaborn scipy jupyter`

## 1.2 必装库及用途

| 库 | 用途 |
|----|------|
| numpy | 多维数组、向量化运算、线性代数 |
| pandas | 表格与时间序列、读写、分组、合并 |
| matplotlib | 二维绘图（折线、散点、直方图等） |
| seaborn | 统计图、热力图、多序列对比 |
| scipy | 统计检验、优化、特殊函数 |
| jupyter | 交互式 Notebook，便于探索与笔记 |

安装后可在 Python 中验证：

```python
import numpy as np
import pandas as pd
print(np.__version__, pd.__version__)  # 例如 1.24.3 2.0.0
```

## 1.3 Jupyter Notebook / JupyterLab

- **Notebook**：以「单元格」为单位运行代码或撰写 Markdown；适合**逐步探索数据、保留中间结果与图表、写说明与公式**。  
- **JupyterLab**：在 Notebook 基础上提供多标签、文件树、终端等，界面更接近 IDE。  
启动方式：在终端中进入项目目录后执行 `jupyter notebook` 或 `jupyter lab`，浏览器会打开对应界面。  
建议：Phase 0 的「数据加载 → 清洗 → 统计 → 可视化」练习在 **Notebook** 中完成，便于保存过程与图表，并与概率统计笔记对照。

## 本章小结（第一章）

**核心概念**：Python 版本与发行版（Anaconda/Miniconda）；环境隔离（conda 环境）；必装库及其用途；Jupyter Notebook/Lab。

**主要结论**：  
1. 建议 Python 3.9+；`conda create -n quant python=3.10` 创建量化环境。  
2. 必装：numpy、pandas、matplotlib、seaborn、scipy、jupyter；验证用 `import` 与 `__version__`。  
3. Jupyter 以单元格为单位运行代码与 Markdown，适合数据探索与笔记。

---

# 第二章 NumPy 基础

**动机**：收益序列、因子矩阵、权重向量、协方差矩阵都是多维数值数据；NumPy 的 ndarray 提供同类型、连续存储、向量化运算，比纯 Python 循环快一个数量级以上，是 Pandas 与科学计算的基础。

## 2.1 ndarray 与创建

### 2.1.1 什么是 ndarray

**ndarray**（N-dimensional array）是 NumPy 的核心对象：**同类型**元素的**多维数组**，在内存中**连续存储**，支持**向量化**运算（对整个数组或按轴运算，无需显式循环），因此比纯 Python 列表循环快一个数量级以上。  
量化中：**收益序列**、**因子矩阵**、**权重向量**、**协方差矩阵**等，都常用 ndarray 表示。

### 2.1.2 从列表创建

```python
import numpy as np

# 一维
a = np.array([1, 2, 3, 4, 5])
print(a)           # [1 2 3 4 5]
print(type(a))     # <class 'numpy.ndarray'>

# 二维（矩阵）
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]
```

**注意**：列表中的子列表长度必须一致，否则得到的是「对象数组」而非规则矩阵，后续运算会受限。

### 2.1.3 常用创建函数

**zeros / ones**：全 0 或全 1，需指定形状。

```python
np.zeros(5)              # 一维 5 个 0
np.zeros((2, 3))        # 2 行 3 列
np.ones((2, 3))         # 2 行 3 列全 1
np.ones((2, 3), dtype=np.int32)  # 指定类型
```

**arange**：类似内置 `range`，但得到数组；**左闭右开**。

```python
np.arange(10)           # [0,1,...,9]
np.arange(2, 10, 2)      # [2,4,6,8]，步长 2
np.arange(0, 1, 0.1)     # 浮点需注意精度
```

**linspace**：**闭区间**上**等分**成 n 个点，适合做横轴。

```python
np.linspace(0, 1, 5)     # [0, 0.25, 0.5, 0.75, 1]
np.linspace(0, 1, 5, endpoint=False)  # 不含右端点，等价于 arange(0,1,0.2)
```

**随机数**：

```python
np.random.rand(3, 2)     # [0,1) 均匀，形状 (3,2)
np.random.randn(5)       # 标准正态 N(0,1)，5 个数
np.random.randn(2, 3)    # 形状 (2,3)
```

**说明**：`np.random` 在新版 NumPy 中建议使用 `np.random.default_rng()` 生成器，此处仍用常见写法以便与现有资料一致；生产代码可改为显式生成器。

---

## 2.2 形状、维度与 dtype

### 2.2.1 shape、ndim、size

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)   # (2, 3)，即 2 行 3 列
print(a.ndim)   # 2，二维
print(a.size)   # 6，元素总数
```

**reshape**：在**不改变元素总数**的前提下改变形状（按行优先填充）。

```python
b = np.arange(6).reshape(2, 3)   # [[0,1,2],[3,4,5]]
c = np.arange(6).reshape(3, 2)   # [[0,1],[2,3],[4,5]]
# reshape(-1, 1) 表示「列数为 1，行数自动推断」
```

**ravel**：展平为一维（默认行优先）；**T** 或 **.transpose()**：转置。

### 2.2.2 dtype 与类型转换

**dtype** 决定元素如何存储与解释（如 int32、float64）。  
创建数组时若列表中含浮点数，通常得到 float64；纯整数为 int32 或 int64（视系统而定）。

```python
a = np.array([1, 2, 3])
print(a.dtype)   # int32 或 int64
b = np.array([1.0, 2, 3])
print(b.dtype)   # float64
```

**类型转换**：

```python
a = np.array([1, 2, 3], dtype=np.float64)
# 或事后转换
a = a.astype(np.float64)
```

**注意**：整数除法在 NumPy 中仍为整数（如 3/2=1）；若需浮点结果，先将数组转为 float 或使用 `np.true_divide`。

---

## 2.3 索引与切片

### 2.3.1 一维

与 Python 列表类似：**从 0 开始**；**负索引**表示从末尾数；**切片 [start:stop:step]**，左闭右开。

```python
a = np.array([10, 20, 30, 40, 50])
print(a[0], a[-1])       # 10, 50
print(a[1:4])            # [20, 30, 40]
print(a[::2])            # [10, 30, 50]，步长 2
```

**注意（视图与副本）**：NumPy 切片得到的是**原数组的视图**（共享内存），修改视图会改变原数组；若需独立副本，应显式 `a[1:4].copy()`。例如 `b = a[1:4]; b[0]=999` 会同时修改 `a[1]`。

### 2.3.2 多维

用**逗号分隔**各维：`a[i, j]` 表示第 i 行第 j 列（均从 0 开始）。  
**切片**：`a[i1:i2, j1:j2]`；某维取「全部」用冒号 `:`。

```python
b = np.arange(12).reshape(3, 4)   # 3 行 4 列
print(b[1, 2])      # 第 2 行第 3 列：6
print(b[1, :])      # 第 2 行整行
print(b[:, 2])      # 第 3 列整列
print(b[0:2, 1:4])  # 前 2 行、第 2 到 4 列的子块
```

### 2.3.3 布尔索引

用**布尔数组**做索引：选出对应位置为 True 的元素，**结果为一维**。

```python
a = np.array([1, -2, 3, -4, 5])
print(a[a > 0])     # [1, 3, 5]
print(a[a < 0])     # [-2, -4]
```

**多条件**：用 `&`（与）、`|`（或）、`~`（非）；**必须加括号**，因为运算符优先级。

```python
print(a[(a > 0) & (a < 4)])   # [1, 3]
print(a[(a < -1) | (a > 4)])  # [-2, -4, 5]
```

**二维**：布尔数组形状需与数组一致或可广播；常用于按条件选行。

```python
b = np.array([[1, 2], [3, 4], [5, 6]])
mask = b[:, 0] > 2    # 第一列大于 2 的行
print(b[mask])        # [[3,4],[5,6]]
```

### 2.3.4 花式索引（整数数组索引）

用**整数数组**指定要取的下标，可重复、可乱序。

```python
a = np.array([10, 20, 30, 40, 50])
print(a[[0, 2, 4]])   # [10, 30, 50]
print(a[[4, 0, 2]])   # [50, 10, 30]
```

多维时可用多个整数数组或与切片组合；花式索引得到的是**副本**而非视图。

---

## 2.4 运算与广播

### 2.4.1 逐元素运算

同形状数组的 `+`、`-`、`*`、`/`、`**` 均为**逐元素**运算；标量与数组运算时，标量作用于每个元素。

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)    # [5, 7, 9]
print(a * b)    # [4, 10, 18]
print(a * 2)    # [2, 4, 6]
```

**常用数学函数**：`np.sqrt`、`np.exp`、`np.log`、`np.abs`、`np.maximum` 等，均为逐元素。

### 2.4.2 矩阵乘法与内积

**矩阵乘法**：`A @ B` 或 `np.dot(A, B)`。  
- 二维 × 二维：标准矩阵乘法，(m×n)(n×p)=m×p。  
- 一维 × 一维：**内积**（标量）。  
- 二维 × 一维：矩阵乘向量，得一维。

```python
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 1])
print(A @ v)        # [3, 7]
print(v @ v)        # 2，内积
```

### 2.4.3 约简（聚合）

**sum、mean、std、var、min、max** 等可对**整个数组**或**按轴**计算。  
**axis**：沿哪一维聚合，该维「消失」；**axis=0** 沿行（按列聚合），**axis=1** 沿列（按行聚合）。

```python
b = np.arange(12).reshape(3, 4)
print(b.sum())           # 全体和：66
print(b.sum(axis=0))     # 每列的和，形状 (4,)
print(b.sum(axis=1))     # 每行的和，形状 (3,)
print(b.mean(axis=0))    # 每列均值
```

**argmax、argmin**：最大/最小值的**下标**（扁平或按轴）。

### 2.4.4 广播（Broadcasting）

当两个数组形状**不完全相同**但满足一定规则时，NumPy 会自动将形状「扩展」以便逐元素运算，这一机制称为**广播**。

**规则**：从**最右端**维度对齐；若某维**长度相等**或**其一为 1**，则可广播；否则报错。

**例**：  
- (3, 4) 与 (4,)：从右对齐，(4,) 视为 (1, 4)，再扩展为 (3, 4)，可相加。  
- (3, 4) 与 (3, 1)：后者扩展为 (3, 4)，可相加。  
- (3, 4) 与 (3,)：从右对齐，(3,) 与 4 不对齐且都非 1，**不能**直接相加；需把 (3,) 改为 (3, 1) 再与 (3, 4) 运算。

```python
A = np.ones((3, 4))
v = np.array([1, 2, 3, 4])
print((A + v).shape)   # (3, 4)，v 被当作 (1,4) 再扩展
print(A + np.array([[1], [2], [3]]).shape)  # (3, 4)
```

**常见用法**：每列减去列均值、每行除以行和等，都依赖广播。

**注意（广播常见错误）**：形状 (3,) 与 (3,4) 不能直接逐元素运算，因为从右对齐时 (3,) 与 4 既不相等也不为 1。应先把 (3,) 变为列向量再广播，例如 `v.reshape(-1, 1) + A` 或 `v[:, np.newaxis] + A`。

---

## 2.5 常用函数小结与量化中的用法

- **创建**：`array`、`zeros`、`ones`、`arange`、`linspace`、`random.randn/rand`  
- **形状**：`shape`、`reshape`、`ravel`、`T`  
- **统计**：`sum`、`mean`、`std`、`var`、`min`、`max`、`argmax`、`argmin`  
- **线性代数**：`np.dot`、`np.linalg.inv`、`np.linalg.eig`、`np.linalg.svd`（后续阶段会用到）

在量化中：**收益序列**、**因子矩阵**、**权重向量**多为 ndarray；**向量化**可避免循环，提高回测与因子计算速度；**协方差矩阵、特征值分解**等也依赖 NumPy 的线性代数模块。

## 本章小结（第二章）

**核心概念**：ndarray（同类型、多维、向量化）；shape、reshape、axis；索引与切片（视图 vs 副本）；布尔索引与花式索引；逐元素运算与矩阵乘法；广播规则。

**主要结论**：  
1. 创建：`np.array`、`zeros`、`ones`、`arange`、`linspace`、`random.randn`。  
2. 聚合按轴：`axis=0` 沿行（按列），`axis=1` 沿列（按行）；矩阵乘 `A @ B`。  
3. 广播：从最右维对齐，维长相等或其一为 1 则可广播；形状 (3,) 与 (3,4) 不能直接相加，需 `reshape(-1,1)`。

---

# 第三章 Pandas 基础

**动机**：量化数据多为带标签的表格（日期、股票代码、开高低收、因子、收益）；Pandas 的 Series/DataFrame 提供行索引与列名、缺失值处理、分组聚合、时间序列重采样等，是回测与因子分析的主要数据结构。

## 3.1 Series

### 3.1.1 概念与创建

**Series** 是**带标签的一维数组**：除了一列数据外，还有与之对应的**索引**（index）。  
创建方式：`pd.Series(data, index=...)`。若只传 data（如列表），则默认索引为 0,1,2,…。

```python
import pandas as pd

s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)
# a    1
# b    2
# c    3
# d    4
# dtype: int64
```

**属性**：`s.values` 得到底层 NumPy 数组；`s.index` 得到索引；`s.dtype` 得到数据类型。

### 3.1.2 索引与切片

可按**标签**或**位置**访问；切片时**按标签**是闭区间，**按位置**仍是左闭右开。

```python
print(s['b'])      # 2
print(s[1])        # 2，位置
print(s['a':'c'])  # 含 a,b,c（标签切片闭区间）
print(s.iloc[0:2]) # 位置切片，前两个
```

**布尔索引**：`s[s > 2]` 得到大于 2 的子 Series。

### 3.1.3 常用方法

- **聚合**：`s.sum()`、`s.mean()`、`s.std()`、`s.min()`、`s.max()`  
- **描述**：`s.describe()` 返回 count、mean、std、min、25%、50%、75%、max  
- **缺失值**：`s.dropna()` 删除缺失；`s.fillna(0)` 用 0 填充缺失

---

## 3.2 DataFrame

### 3.2.1 概念与创建

**DataFrame** 是**二维表格**：每列是一个 **Series**，有列名；行有**行索引**。  
创建方式：字典（键为列名）、二维数组+列名、或从文件读取。

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}, index=['x', 'y', 'z'])
print(df)
#    A  B  C
# x  1  4  7
# y  2  5  8
# z  3  6  9
```

**属性**：`df.shape`、`df.columns`、`df.index`、`df.dtypes`；`df.head()`、`df.tail()` 查看前/后几行。

### 3.2.2 索引与选择

**列**：  
- `df['A']` 得到一列，类型为 Series；  
- `df[['A', 'C']]` 得到两列，类型为 DataFrame。

**行**：  
- **按标签**：`df.loc['x']`、`df.loc['x':'y']`（闭区间）；  
- **按位置**：`df.iloc[0]`、`df.iloc[0:2]`（左闭右开）。  
**行列同时**：`df.loc['x', 'A']`、`df.iloc[0, 1]`、`df.loc['x':'y', ['A', 'B']]`。

**布尔索引**：常用于按条件选行。

```python
df[df['A'] > 1]           # A 列大于 1 的行
df[(df['A'] > 1) & (df['B'] < 6)]  # 多条件，注意括号
```

**注意（链式索引与 SettingWithCopyWarning）**：链式索引 `df['A'][0] = 1` 或 `df[df['A']>1]['B'] = 0` 可能触发警告或赋值无效，因为中间结果可能是副本。推荐用 `.loc`/`.iloc` 一次性定位并赋值，例如 `df.loc[df['A']>1, 'B'] = 0`。

---

## 3.3 读写与缺失值

### 3.3.1 读取 CSV

```python
df = pd.read_csv('path/to/file.csv', encoding='utf-8')
# 常用参数：
# sep=','          分隔符
# header=0        第几行作列名
# index_col=0     第几列作行索引
# parse_dates=['date']  将指定列解析为日期
# na_values=['', 'NA'] 将哪些值视为缺失
```

**编码**：中文或特殊字符时常用 `encoding='utf-8'` 或 `'gbk'`。

### 3.3.2 写入

```python
df.to_csv('output.csv', index=False)  # 不写行索引
df.to_parquet('output.parquet')      # 列式存储，大表更高效
```

### 3.3.3 缺失值

- **检测**：`df.isna()` 返回同形状的布尔 DataFrame；`df.isna().sum()` 每列缺失个数。  
- **删除**：`df.dropna(axis=0, how='any')` 删除含缺失的行；`how='all'` 仅当整行全为缺失时删。  
- **填充**：`df.fillna(0)`、`df.fillna(df.mean())`（每列用该列均值填）。  
量化中：停牌、未上市等会产生 NaN；在算收益、因子前需**统一处理**（删或填），否则后续运算会传播 NaN。

---

## 3.4 合并与拼接

### 3.4.1 concat

**按行堆叠**（上下拼接）：`pd.concat([df1, df2], axis=0)`。  
**按列堆叠**（左右拼接）：`pd.concat([df1, df2], axis=1)`。  
`join='inner'` 或 `'outer'` 控制索引对齐方式（取交或并）。

### 3.4.2 merge

按**列（键）**连接两张表，类似 SQL 的 JOIN。

```python
pd.merge(left, right, on='key', how='left')   # 左连接
# how 可选 'left','right','inner','outer'
# 若键名不同，用 left_on=..., right_on=...
```

量化中：常把「因子表」与「收益表」按 **日期、股票代码** merge，得到对齐后的面板数据。

---

## 3.5 分组与聚合

### 3.5.1 groupby 基本用法

按某列（或若干列）**分组**，再对每组做**聚合**（如 sum、mean、std）。

```python
# 按 'symbol' 分组，对 'ret' 求均值
df.groupby('symbol')['ret'].mean()

# 对多列聚合，且用不同函数
df.groupby('sector').agg({'ret': ['mean', 'std'], 'volume': 'sum'})
```

**多列分组**：`df.groupby(['date', 'sector'])['ret'].mean()` 得到多级索引的 Series。  
在量化中：**按行业、按日期的收益统计**、**因子分层收益**，都依赖 groupby。

### 3.5.2 时间序列相关

**日期索引**：`pd.to_datetime(df['date'])`；`df.set_index('date')` 将日期设为索引。  
**重采样**：`df.resample('M').mean()` 按月聚合；`'D'`、`'W'`、`'Q'` 等表示日、周、季。  
**滞后**：`df['ret'].shift(1)` 得到滞后一期的序列；用于计算收益或动量。  
**滚动**：`df['ret'].rolling(window=20).mean()` 20 日移动平均；`.std()` 为滚动标准差（如滚动波动率）。

---

## 3.6 描述统计与简单检验

- **描述**：`df.describe()` 给出每列的 count、mean、std、min、25%、50%、75%、max。  
- **单列**：`df['ret'].mean()`、`.std()`、`.skew()`、`.kurt()`。  
- **与 SciPy 结合**：  
  - 单样本 t 检验（H₀: 均值为 0）：`from scipy import stats; stats.ttest_1samp(df['ret'], 0)`  
  - Pearson 相关系数与 p 值：`stats.pearsonr(df['x'], df['y'])`  
  对应数理统计中的 t 检验与相关系数检验，便于在数据上验证所学内容。

## 本章小结（第三章）

**核心概念**：Series（带标签一维）、DataFrame（二维表格）；`.loc`（标签）与 `.iloc`（位置）；读写、缺失值、合并、分组聚合；时间序列（to_datetime、resample、shift、rolling）。

**主要结论**：  
1. 索引：`df.loc['行','列']`、`df.iloc[i,j]`；避免链式索引导致 SettingWithCopyWarning，用 `.loc` 一次性赋值。  
2. 读写：`read_csv`、`to_csv`；缺失值 `isna`、`dropna`、`fillna`；合并 `concat`、`merge`；分组 `groupby().agg()`。  
3. 时间：`shift(1)` 滞后一期；`rolling(window).mean()` 滚动均值（如 20 日均线）。

---

# 第四章 Matplotlib 基础

**动机**：回测需绘制收益曲线、回撤、分布直方图、IC 序列等；Matplotlib 提供底层绘图控制，面向对象接口（fig, ax）便于多子图与复用，是 Seaborn 的基础。

## 4.1 两种风格

- **pyplot 接口**：`import matplotlib.pyplot as plt`；`plt.plot(x, y)`、`plt.xlabel(...)`、`plt.show()`。适合快速出图。  
- **面向对象**：先创建图形与坐标轴 `fig, ax = plt.subplots()`，再 `ax.plot(x, y)`、`ax.set_xlabel(...)`。便于**多子图、精细控制**，推荐在 Notebook 和脚本中**统一使用面向对象**。

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3], [2, 4, 3], label='line1')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Title')
ax.legend()
plt.tight_layout()
plt.show()
```

## 4.2 常用图类型

- **折线**：`ax.plot(x, y)`；多条线可多次 `plot` 或传入多列。  
- **散点**：`ax.scatter(x, y, c=..., s=...)`；`c`、`s` 可表示第三维或大小。  
- **直方图**：`ax.hist(data, bins=30)`，看分布形状。  
- **条形**：`ax.bar(x, height)`、`ax.barh(y, width)`。

## 4.3 子图与保存

```python
fig, axes = plt.subplots(2, 2)   # 2×2 子图
axes[0, 0].plot(...)
axes[0, 1].scatter(...)
# 或遍历 axes.flat
fig.savefig('fig.png', dpi=150)
```

量化中：收益曲线、回撤曲线、因子分组收益柱状图、收益分布直方图、IC 时间序列图等，都由这些基本元素组合完成。

## 本章小结（第四章）

**核心概念**：pyplot 与面向对象两种风格；figure 与 axes；常用图类型（折线、散点、直方图、条形）；子图与保存。

**主要结论**：  
1. 推荐面向对象：`fig, ax = plt.subplots()`，`ax.plot`、`ax.set_xlabel`、`ax.legend`。  
2. 常用：`ax.plot`、`ax.scatter`、`ax.hist`、`ax.bar`；多子图 `subplots(2,2)`；保存 `fig.savefig('fig.png', dpi=150)`。

---

# 第五章 Seaborn 进阶可视化

**动机**：与 DataFrame 列名直接配合、按列分组着色、统一主题，可快速得到报告级图表；热力图、箱线图等统计图在因子分析中常用。

- **与 DataFrame 配合**：`sns.lineplot(data=df, x='date', y='ret', hue='symbol')`，按 `hue` 分组着色。  
- **常用图**：`sns.lineplot`、`sns.scatterplot`、`sns.regplot`（带回归线）、`sns.histplot`、`sns.kdeplot`、`sns.boxplot`、`sns.violinplot`、`sns.heatmap`（相关矩阵、IC 矩阵等）。  
- **主题**：`sns.set_theme()`、`sns.set_style('darkgrid')` 等，可统一风格。  
量化报告与复现中，用 Seaborn 可快速得到风格一致的多图。

## 本章小结（第五章）

**核心概念**：Seaborn 与 DataFrame 的列名配合；hue 分组着色；常用统计图。

**主要结论**：  
1. `sns.lineplot(data=df, x='date', y='ret', hue='symbol')` 按 symbol 分组画线。  
2. 常用：`lineplot`、`scatterplot`、`regplot`、`histplot`、`kdeplot`、`boxplot`、`heatmap`（相关矩阵、IC 矩阵）。

---

# 第六章 Phase 0 综合练习建议

1. **数据**：用 `pd.read_csv` 读入一段股票或指数日线（开、高、低、收、成交量）；若无现成数据，可用 `yfinance` 或 `akshare` 获取，或构造简单模拟数据。  
2. **清洗与计算**：计算日收益率（如 (close/close.shift(1)−1)）；用 `dropna` 或 `fillna` 处理缺失；计算 5 日、20 日简单移动平均。  
3. **统计与检验**：用 `describe()`、`mean()`、`std()` 看收益与波动；用 `scipy.stats.ttest_1samp(收益, 0)` 检验均值是否显著为 0；若有两只标的，用 `pearsonr` 看两列收益的相关性。  
4. **可视化**：用 Matplotlib 或 Seaborn 画：(1) 收益序列折线；(2) 收益直方图；(3) 滚动 20 日波动率曲线；(4) 若有多标的，画相关性热力图。  
这样把**概率统计**（均值、方差、t 检验、相关系数）与 **Python 量化栈**（NumPy/Pandas/绘图）串起来，为 Phase 1 的时间序列与回归打基础。

**例 6.1（综合例题：模拟日收益的统计与检验）**  
**题干**：用 NumPy 生成 252 个服从 N(0.0005, 0.0001) 的「日收益率」样本（即 μ=0.05%、σ=1%），用 Pandas 存为 Series，计算样本均值、样本标准差、均值的 95% 置信区间（t 区间），并做 H₀: μ=0 的 t 检验；最后用 Matplotlib 画收益序列与直方图。

**解答（可运行代码）**：

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
n = 252
mu_true, var_true = 0.0005, 0.0001
rets = np.random.normal(mu_true, np.sqrt(var_true), n)
s = pd.Series(rets, name='ret')

xbar = s.mean()
s_std = s.std(ddof=1)
print('样本均值 x̄ =', round(xbar, 6), '  样本标准差 s =', round(s_std, 6))

# 95% 置信区间 (t)
t_half = stats.t.ppf(0.975, n - 1)
margin = t_half * s_std / np.sqrt(n)
ci = (xbar - margin, xbar + margin)
print('μ 的 95% 置信区间:', [round(ci[0], 6), round(ci[1], 6)])

# t 检验 H0: mu=0
t_stat, p_value = stats.ttest_1samp(s, 0)
print('t 统计量 =', round(t_stat, 4), '  p 值 =', round(p_value, 4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(s.values)
axes[0].set_title('日收益序列')
axes[0].set_xlabel('交易日')
axes[1].hist(s, bins=25, edgecolor='black', alpha=0.7)
axes[1].set_title('收益直方图')
plt.tight_layout()
plt.show()
```

**结果说明**：每次运行因随机数不同，x̄、s、置信区间和 p 值会略有变化；若 μ_true=0.0005，多数情况下置信区间会包含 0.0005，且 p 值可能大于或小于 0.05（取决于样本）。直方图应大致呈钟形，与正态假设一致。  
**单位**：收益为小数（0.0005 表示 0.05%）；若需百分数显示，可将 `rets*100` 再算。

## 本章小结（第六章）

**核心概念**：Phase 0 综合流程（数据读入→清洗→收益与指标→统计检验→可视化）；与概率论、数理统计的对应（样本均值、S²、t 区间、t 检验、相关系数）。

**主要结论**：  
1. 流程：`read_csv` → 收益 `(close/close.shift(1)-1)`、`rolling(20).mean()` → `describe`、`ttest_1samp`、`pearsonr` → 折线/直方/滚动波动/热力图。  
2. 例 6.1 完整实现：模拟 252 日收益 → 样本均值、S、95% t 置信区间、H₀: μ=0 的 t 检验 → 序列图与直方图。

**学习建议**：每节代码在 Notebook 中**亲手敲一遍**并观察输出；索引与广播部分可多试几种形状，避免只记结论；遇到报错时先看**形状是否匹配、是否有 NaN**，再查文档。学完本章应能独立完成「读入 CSV → 清洗 → 算收益与简单指标 → 做 t 检验与相关 → 画 2～3 种图」的完整流程。
