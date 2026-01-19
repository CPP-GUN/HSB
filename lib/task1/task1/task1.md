# Q1（问题一）建模/代码交付文档（供论文手编写用）

> **定位**：这是“建模手 + 代码手”内部交付给“论文手”的技术说明文档。
>
> **目标**：把 Q1 的**模型定义、公式推导、实现口径、输出文件含义、结果解释边界**写清楚，确保论文手能够：
>
> 1. 在正文中严格复述模型；2) 在结果段落中解释“为什么会这样”；3) 能从 `outputs/` 直接取数写表/画图。
>
> **注意**：本文档**以现有可复现代码与运行输出为准**，不追溯旧版本“63 对 / m=3 / 美第一”等历史口径。

---

## 0. 代码与数据的唯一事实来源

### 0.1 目录与入口

* 入口：`mainpip.py`
* 模块：

  * `task1_correlation_cluster.py`：相关性矩阵 + 强相关对 + 层次聚类（Average Linkage）
  * `task1_pca_importance.py`：PCA + 主成分选择 + 要素重要性
  * `task1_evaluation.py`：综合得分 + 归一化 + 排名
  * `task1_utils.py`：以上所有数学函数实现

### 0.2 输入数据

* `data_standardized.csv`

  * 第 1 列：国家（Country）
  * 后续列：指标（Indicators）
  * **现行口径**：`p = 指标列总数`（以 CSV 实际列为准，当前为 24 指标）
  * 样本量：`n = 国家数`（当前为 10 国）

> 重要：所有模型计算都直接基于 `data_standardized.csv` 的指标列。指标数与列名变化会导致相关对数、PCA 贡献率、排名等改变，这是正常现象。

---

## 1. 模型流程总览（写论文可用的“方法流程”）

总体流程：

1. **读取标准化数据矩阵** $X' \in \mathbb{R}^{n\times p}$
2. **相关性分析**：计算 Pearson 相关矩阵 $R\in\mathbb{R}^{p\times p}$，提取强相关边集 $\mathcal{E}$（$|r|>0.7$）
3. **层次聚类**：以 $d_{jk}=1-|r_{jk}|$ 构建距离矩阵 $D$，Average Linkage 得到聚类 linkage 矩阵 $Z$
4. **PCA**：对 $X'$ 做主成分分解，得到方差贡献率 $\eta_k$、主成分得分 $PC_{ik}$ 与载荷 $l_{jk}$
5. **关键要素识别**：按 $I_j=\sum_{k=1}^{m} l_{jk}^2\eta_k$ 计算要素重要性
6. **综合评估**：按 $S_i=\sum_{k=1}^{m} w_k PC_{ik}$ 得到国家综合得分，并做样本内归一化与排名

对应输出文件：见第 6 节。

---

## 2. 数据表示与预处理口径（对齐实现）

### 2.1 数据矩阵

从 `data_standardized.csv` 读取：

* 国家集合：${c_1,\ldots,c_n}$
* 指标集合：${f_1,\ldots,f_p}$
* 数据矩阵：

$$
X' = \begin{bmatrix}
x'*{11} & x'*{12} & \cdots & x'*{1p}\
x'*{21} & x'*{22} & \cdots & x'*{2p}\
\vdots & \vdots & \ddots & \vdots\
x'*{n1} & x'*{n2} & \cdots & x'_{np}
\end{bmatrix}
$$

其中 $x'_{ij}\in[0,1]$ 为 Min–Max 标准化后的值。

### 2.2 标准化说明

现阶段**不在代码中重复标准化**，原因：

* `data_standardized.csv` 已满足 $x'\in[0,1]$
* 相关性与 PCA 输入均为该矩阵

如论文手需要描述标准化，可写（对应 Min–Max）：

$$
x'*{ij} = \frac{x*{ij}-\min_j}{\max_j-\min_j}
$$

---

## 3. 相关性分析（Pearson）

### 3.1 Pearson 相关系数定义

对任意两个指标 $f_j,f_k$，Pearson 相关系数：

$$
r_{jk} = \frac{\sum_{i=1}^{n}(x'*{ij}-\bar{x}'*j)(x'*{ik}-\bar{x}'*k)}
{\sqrt{\sum*{i=1}^{n}(x'*{ij}-\bar{x}'*j)^2},\sqrt{\sum*{i=1}^{n}(x'_{ik}-\bar{x}'_k)^2}}
$$

其中：

$$
\bar{x}'*j=\frac{1}{n}\sum*{i=1}^{n}x'_{ij}
$$

### 3.2 相关矩阵

$$
R = [r_{jk}]_{p\times p}
$$

性质：

* $r_{jj}=1$
* $r_{jk}=r_{kj}$
* $-1\le r_{jk}\le 1$

### 3.3 强相关边集（用于弦图/网络图/机制提取）

判据（实现口径）：

$$
|r_{jk}|>0.7
$$

强相关对数量：

$$
N_{strong} = \sum_{1\le j<k\le p} \mathbb{1}(|r_{jk}|>0.7)
$$

**实现细节（对应代码）**：

* 只统计上三角（$j<k$）避免重复
* 输出三元组：$(f_j,f_k,r_{jk})$

> 说明给论文手：
>
> * “强相关对数量”不是常数，随指标集与数据版本变化。
> * 当前运行结果显示 $N_{strong}=86$（由 `strong_correlations.csv` 行数给出）。

---

## 4. 层次聚类（Average Linkage）

> 目的：从相关结构出发，获得指标的“自然分组”，为论文中的簇解释（如人才—研发、基础设施簇等）提供依据。

### 4.1 距离度量

按文档与实现口径，定义指标间距离：

$$
d_{jk}=1-|r_{jk}|
$$

因此距离矩阵：

$$
D=[d_{jk}]_{p\times p}
$$

性质：

* $d_{jj}=0$
* $d_{jk}\in[0,1]$
* 相关越强（$|r|$ 越大），距离越小

### 4.2 Average Linkage（平均连接）

设两个簇为 $C_a,C_b$，簇间距离定义为：

$$
D(C_a,C_b)=\frac{1}{|C_a||C_b|}\sum_{x\in C_a}\sum_{y\in C_b} d(x,y)
$$

### 4.3 输出 linkage 矩阵

层次聚类算法输出 linkage 矩阵 $Z$（SciPy 标准格式），每行表示一次合并操作：

* 合并的两个簇（或叶子）编号
* 合并距离（高度）
* 新簇大小

> 给论文手的用法：
>
> * 用 $Z$ 可直接画 dendrogram
> * 或用 `cluster_distance_matrix.csv` + `cluster_linkage.csv` 生成“聚类热力图”

---

## 5. PCA（主成分分析）与关键要素识别

> 目的：把多指标信息压缩为少数综合因子（主成分），并借助载荷结构评估“关键驱动要素”。

### 5.1 数据中心化（理论描述）

尽管实现中由 PCA 内部完成中心化，论文可按标准写：

$$
\tilde{X}=X'-\bar{X}'
$$

其中 $\bar{X}'$ 为按列均值组成的向量。

### 5.2 协方差矩阵与特征分解（理论描述）

$$
C=\frac{1}{n-1}\tilde{X}^T\tilde{X}
$$

特征分解：

$$
C=V\Lambda V^T
$$

* $V=[\mathbf{v}_1,\ldots,\mathbf{v}_p]$ 为特征向量
* $\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_p)$ 为特征值

### 5.3 方差贡献率

$$
\eta_k=\frac{\lambda_k}{\sum_{i=1}^{p}\lambda_i}
$$

累积贡献率：

$$
\eta_{cum}(m)=\sum_{k=1}^{m}\eta_k
$$

### 5.4 主成分选择准则（实现口径）

选择最小 $m$ 满足：

$$
\eta_{cum}(m)\ge 0.85
$$

实现等价：

* 在 `pca_variance.csv` 中寻找第一条 `Cumulative_Ratio >= 0.85` 的主成分编号
* 当前运行结果为：**m = 4**

> 给论文手的解释句：
> “在当前指标集与数据版本下，前三个主成分累计贡献率未达到 0.85，因此依据准则取 m=4。”

### 5.5 主成分得分

主成分得分（国家在主成分上的坐标）：

$$
PC_{ik}=\sum_{j=1}^{p} v_{jk},\tilde{x}_{ij}
$$

实现中由 `sklearn.decomposition.PCA().fit_transform(X')` 得到得分矩阵 $Z$。

### 5.6 载荷矩阵（实现口径）

载荷可按特征向量矩阵表示：

$$
L=[l_{jk}] = V
$$

在代码输出中：`pca_loadings.csv` 存储 $p\times p$ 的载荷矩阵（列为 PC1..PCp）。

> 注：不同教材对“载荷”定义存在 $v_{jk}$ 与 $\sqrt{\lambda_k}v_{jk}$ 两种口径。本项目现行实现采用 **$l_{jk}=v_{jk}$**，并与后续重要性公式一致。

### 5.7 要素重要性（关键驱动要素）

按文档与实现口径：

$$
I_j = \sum_{k=1}^{m} l_{jk}^2,\eta_k
$$

解释：

* $l_{jk}^2$ 衡量要素 $j$ 在第 $k$ 个综合因子上的贡献权重
* $\eta_k$ 衡量该综合因子的解释力度
* 因此 $I_j$ 是“跨主成分的综合重要性”

派生量（实现输出）：

* 归一化重要性：

$$
I'*j=\frac{I_j}{\sum*{j=1}^{p}I_j}
$$

* 相对贡献度（以最大者为 100%）：

$$
Contribution_j=\frac{I_j}{\max(I)}\times 100%
$$

输出文件：`factor_importance.csv`

---

## 6. 综合评估模型（国家得分与排名）

> 目的：用 PCA 提取的综合因子对各国做样本内相对评估。

### 6.1 权重计算

对选取的 $m$ 个主成分，按贡献率归一化作为权重：

$$
w_k = \frac{\eta_k}{\sum_{j=1}^{m}\eta_j},\quad \sum_{k=1}^{m}w_k=1
$$

### 6.2 综合得分

国家 $i$ 的综合得分：

$$
S_i = \sum_{k=1}^{m} w_k,PC_{ik}
$$

实现：`S = Z[:,:m] @ w`。

### 6.3 样本内归一化（重要：解释“100%”现象）

为便于展示，采用 Min–Max 将得分映射到 $[0,1]$：

$$
S'*i=\frac{S_i-S*{\min}}{S_{\max}-S_{\min}}\in[0,1]
$$

因此必然满足：

* $\max_i S'_i=1$（样本内第一名显示为 1 或 100%）
* $\min_i S'_i=0$（样本内最后一名显示为 0）

> 给论文手的关键解释：
> “评分为 1（或 100%）不表示绝对满分，而是样本内相对标尺：在所选 10 国中该国综合得分最高，因此归一化后为 1。”

### 6.4 排名

按 $S'_i$ 降序排名：

$$
Rank(i)=1+\sum_{j\ne i}\mathbb{1}(S'_j>S'_i)
$$

输出文件：`comprehensive_evaluation.csv`。

---

## 7. 输出文件字典（论文手“拿来就用”）

所有输出位于 `outputs/`：

1. `correlation_matrix.csv`

   * 指标×指标的 Pearson 相关矩阵 $R$
   * 用于：相关性热力图（Fig1）

2. `strong_correlations.csv`

   * 三列：`Indicator_1, Indicator_2, Correlation`
   * 用于：强相关边集、弦图（Fig2）、网络图（Fig7/社区）以及机制提取
   * 行数即强相关对数量 $N_{strong}$（当前为 86）

3. `cluster_distance_matrix.csv`

   * 距离矩阵 $D=1-|R|$
   * 用于：聚类热图底层输入

4. `cluster_linkage.csv`

   * linkage 矩阵 $Z$（Average Linkage）
   * 用于：dendrogram / clustermap 的排序依据

5. `pca_variance.csv`

   * 每个 PC 的 `Eigenvalue, Variance_Ratio, Cumulative_Ratio`
   * 用于：方差贡献率柱状+折线（Fig4）以及主成分数 $m$ 的依据

6. `pca_loadings.csv`

   * 载荷矩阵 $L$（行：指标；列：PC）
   * 用于：载荷热图（Fig3）、biplot（Fig5）与解释主成分含义

7. `factor_importance.csv`

   * 每个指标的 `Importance, Normalized, Contribution_%`
   * 用于：要素重要性排序（Fig6）、Top-k 关键要素陈述

8. `comprehensive_evaluation.csv`

   * 国家得分：`Country, Score, Score_Normalized, Rank`
   * 用于：综合得分柱状图（Fig4/Fig9）与表格排名

---

## 8. 与“旧文档结果”不一致的原因（论文手可写“差异说明”）

> 这一节用于论文手解释：为什么强相关对数量、主成分数、排名可能与历史版本不同。

### 8.1 指标维度口径差异

旧文本中存在 21/24 指标混用的痕迹。

* 本项目现行实现：以 `data_standardized.csv` 指标列为准，当前 $p=24$
* 指标数变化会影响：

  * 相关矩阵维度与强相关对数量
  * PCA 贡献率分布与 $m$ 选择
  * 最终权重 $w$ 与综合排名

### 8.2 PCA 选取规则随数据变化

即使规则固定 $\eta_{cum}(m)\ge 0.85$，不同数据版本下 $m$ 也可能变化。

当前版本计算得到 $m=4$，因此所有后续得分均以 4 个主成分为基础。

### 8.3 归一化标尺导致“第一名=1”

由于采用样本内 Min–Max 归一化，第一名必为 1（100%）。这是标尺定义，不代表绝对“满分”。

---

## 9. 论文写作建议（把代码输出映射到论文段落）

### 9.1 方法部分（可直接用）

* 数据：$n=10,p=24$，Min–Max 标准化
* 相关：Pearson，阈值 $|r|>0.7$
* 聚类：$d=1-|r|$，Average Linkage
* PCA：累计贡献率阈值 0.85 选 $m$
* 重要性：$I_j=\sum l_{jk}^2\eta_k$
* 综合：$S_i=\sum w_k PC_{ik}$，Min–Max 得到 $S'_i$ 与 Rank

### 9.2 结果部分（强推荐写法）

* 强相关对数量：引用 `strong_correlations.csv` 行数
* $m$：引用 `pca_variance.csv` 中首次超过 0.85 的 PC 编号
* Top 关键要素：引用 `factor_importance.csv` 的前 k 行
* 排名表：直接引用 `comprehensive_evaluation.csv`

---

## 10. 当前一次运行的可复现现象（供论文手写“示例结果”）

以你当前控制台输出为例：

* Strong correlation pairs: **86**
* Selected principal components: **4**
* 综合排名（归一化后）：中国第 1、美国第 2 ……

> 论文手若要展示百分制：可将 `Score_Normalized` 乘 100 输出为 0–100，但这属于展示层，不改变模型。
