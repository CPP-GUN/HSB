"""
task1_utils.py
-------------------------------------------------
用途：
实现 Task 1 中所有数学模型的底层计算函数，包括：
- 数据读取
- Pearson 相关矩阵
- 强相关对提取
- 层次聚类（d = 1 - |r|, Average Linkage）
- PCA 分析
- 要素重要性计算
- 综合得分与排名

严格对应建模文档第 4–7 节的数学定义。
-------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

DATADIR = os.path.dirname(os.path.abspath(__file__))

def load_data(path=os.path.join(DATADIR, "data_standardized.csv")):
    df = pd.read_csv(path)
    countries = df.iloc[:, 0]
    indicators = df.columns[1:]
    X = df.iloc[:, 1:].values.astype(float)
    return countries, indicators, X


# ---------- 4.1 Pearson 相关性 ----------

def pearson_correlation(X, indicators):
    R = np.corrcoef(X, rowvar=False)
    return pd.DataFrame(R, index=indicators, columns=indicators)


def strong_correlations(R, threshold=0.7):
    pairs = []
    names = R.columns
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = R.iloc[i, j]
            if abs(r) > threshold:
                pairs.append([names[i], names[j], r])
    return pd.DataFrame(pairs, columns=["Indicator_1", "Indicator_2", "Correlation"])


# ---------- 4.2 层次聚类 ----------

def hierarchical_clustering(R):
    D = 1 - np.abs(R.values)

    # 数值层面对称化（防止浮点误差）
    D = (D + D.T) / 2

    np.fill_diagonal(D, 0.0)

    condensed = squareform(D, checks=True)
    Z = linkage(condensed, method="average")

    return pd.DataFrame(D, index=R.index, columns=R.columns), Z



# ---------- 5.1 PCA ----------

def run_pca(X, indicators):
    pca = PCA()
    Z = pca.fit_transform(X)

    eigenvalues = pca.explained_variance_
    eta = pca.explained_variance_ratio_
    cum_eta = np.cumsum(eta)

    variance_df = pd.DataFrame({
        "Eigenvalue": eigenvalues,
        "Variance_Ratio": eta,
        "Cumulative_Ratio": cum_eta
    }, index=[f"PC{i+1}" for i in range(len(eta))])

    loadings = pd.DataFrame(
        pca.components_.T,
        index=indicators,
        columns=[f"PC{i+1}" for i in range(len(eta))]
    )

    return Z, variance_df, loadings


def select_m(variance_df, threshold=0.85):
    return np.argmax(variance_df["Cumulative_Ratio"].values >= threshold) + 1


# ---------- 5.2 要素重要性 ----------

def factor_importance(loadings, variance_df, m):
    eta = variance_df["Variance_Ratio"].iloc[:m].values
    L = loadings.iloc[:, :m].values

    I = (L ** 2) @ eta
    df = pd.DataFrame({"Importance": I}, index=loadings.index)
    df["Normalized"] = df["Importance"] / df["Importance"].sum()
    df["Contribution_%"] = df["Importance"] / df["Importance"].max() * 100
    return df.sort_values("Importance", ascending=False)


# ---------- 7.1 综合评估 ----------

def comprehensive_score(Z, variance_df, m, countries):
    eta = variance_df["Variance_Ratio"].iloc[:m].values
    w = eta / eta.sum()

    S = Z[:, :m] @ w
    S_norm = (S - S.min()) / (S.max() - S.min())

    df = pd.DataFrame({
        "Country": countries,
        "Score": S,
        "Score_Normalized": S_norm
    })

    df["Rank"] = df["Score_Normalized"].rank(
        ascending=False, method="min"
    ).astype(int)

    return df.sort_values("Rank")
