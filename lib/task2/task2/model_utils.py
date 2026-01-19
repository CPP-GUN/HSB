# model_utils.py
# =====================================================
# 问题二：模型工具模块（不涉及文件读写/不涉及可视化）
# 目标：将“熵权法 + TOPSIS + 灰色关联（加权）”封装为可复用函数
# 数据要求：所有指标均按效益型处理（越大越好）
# =====================================================

import numpy as np


def min_max_normalize(X: np.ndarray) -> np.ndarray:
    """
    Min-Max归一化到[0,1]区间
    对应文档公式：
        x' = (x - min) / (max - min)

    参数：
        X: (n, m) 原始指标矩阵
    返回：
        X_norm: (n, m) 归一化矩阵
    """
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    X_norm = np.zeros((n, m), dtype=float)

    for j in range(m):
        x_min = np.min(X[:, j])
        x_max = np.max(X[:, j])
        # 若某列无差异（max==min），归一化后无法区分，按工程稳定性处理为1
        if x_max > x_min:
            X_norm[:, j] = (X[:, j] - x_min) / (x_max - x_min)
        else:
            X_norm[:, j] = 1.0

    return X_norm


def entropy_weight(X_norm: np.ndarray):
    """
    熵权法（Entropy Weight Method）
    对应文档步骤：
      1) 基于归一化矩阵 X_norm 计算比重矩阵 P
      2) 计算信息熵 E_j
      3) 计算信息效用值 d_j = 1 - E_j
      4) 权重 w_j = d_j / sum(d_j)

    参数：
        X_norm: (n, m) 归一化矩阵，元素在[0,1]
    返回：
        weights: (m,) 权重向量
        entropy: (m,) 信息熵
        redundancy: (m,) 信息效用值(冗余度) = 1 - entropy
    """
    X_norm = np.asarray(X_norm, dtype=float)
    n, m = X_norm.shape
    eps = 1e-12

    # 比重矩阵 P
    col_sum = np.sum(X_norm, axis=0) + eps
    P = X_norm / col_sum

    # 信息熵 E_j = -k * sum(p_ij ln p_ij), k = 1/ln(n)
    k = 1.0 / np.log(n)
    entropy = -k * np.sum(P * np.log(P + eps), axis=0)

    # 信息效用值 d_j
    redundancy = 1.0 - entropy

    # 权重 w_j
    weights = redundancy / (np.sum(redundancy) + eps)

    return weights, entropy, redundancy


def topsis(X_norm: np.ndarray, weights: np.ndarray):
    """
    TOPSIS综合评价
    对应文档步骤：
      1) 向量归一化：r_ij = x'_ij / sqrt(sum_i (x'_ij)^2)
      2) 加权：Z_ij = w_j * r_ij
      3) 正负理想解：Z+ = max_i Z_ij, Z- = min_i Z_ij
      4) 距离：D+、D-（欧氏距离）
      5) 相对接近度：C = D- / (D+ + D-)

    参数：
        X_norm: (n, m) 归一化矩阵
        weights: (m,) 熵权法权重
    返回：
        C: (n,) TOPSIS得分
        D_plus: (n,) 到正理想解距离
        D_minus: (n,) 到负理想解距离
    """
    X_norm = np.asarray(X_norm, dtype=float)
    weights = np.asarray(weights, dtype=float)

    eps = 1e-12
    # 向量归一化
    denom = np.sqrt(np.sum(X_norm ** 2, axis=0)) + eps
    R = X_norm / denom

    # 加权标准化矩阵
    Z = R * weights

    # 正负理想解
    Z_plus = np.max(Z, axis=0)
    Z_minus = np.min(Z, axis=0)

    # 距离
    D_plus = np.sqrt(np.sum((Z - Z_plus) ** 2, axis=1))
    D_minus = np.sqrt(np.sum((Z - Z_minus) ** 2, axis=1))

    # 相对接近度
    C = D_minus / (D_plus + D_minus + eps)

    return C, D_plus, D_minus


def grey_relational_analysis(X_norm: np.ndarray, weights: np.ndarray, rho: float = 0.5):
    """
    灰色关联分析（加权版，严格对齐文档）
    对应文档：
      1) 参考序列 X0：各指标最大值
      2) 关联系数 xi_ij
      3) 灰色关联度 gamma_i = sum_j w_j * xi_ij

    参数：
        X_norm: (n, m) 归一化矩阵
        weights: (m,) 熵权法权重（用于加权）
        rho: 分辨系数，文档默认0.5
    返回：
        gamma: (n,) 灰色关联度
    """
    X_norm = np.asarray(X_norm, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # 参考序列（理想国家）
    X0 = np.max(X_norm, axis=0)

    # 绝对差
    Delta = np.abs(X_norm - X0)
    delta_min = np.min(Delta)
    delta_max = np.max(Delta)

    # 关联系数矩阵 xi
    xi = (delta_min + rho * delta_max) / (Delta + rho * delta_max + 1e-12)

    # 加权灰色关联度（严格按文档）
    gamma = np.sum(xi * weights, axis=1)

    return gamma


def rank_desc(values: np.ndarray) -> np.ndarray:
    """
    将数值按“越大越好”转为名次（1为最好）
    返回每个样本的名次，形状(n,)
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values)  # 从大到小
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks
