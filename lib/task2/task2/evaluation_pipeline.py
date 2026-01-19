# evaluation_pipeline.py
# =====================================================
# 问题二：评估流程模块（不涉及可视化）
# 要求：严格对齐建模文档所有环节
#  - 数据标准化（Min-Max）
#  - 熵权法（权重、信息熵、效用值）
#  - TOPSIS（D+、D-、C、排名）
#  - 灰色关联（加权gamma、排名）
#  - Spearman秩相关（含p值）
#  - Borda综合 + 综合得分S = 0.5*C + 0.5*gamma
#  - 等级划分（按阈值）
#  - 敏感性分析（逐指标权重扰动±30/±15/0，共24*5=120次；Range）
#  - 六维度得分（基于X_norm，按维度均值）
# 输出：全部结果写入 outputs/ 目录
# =====================================================

import os
import numpy as np
import pandas as pd

from model_utils import (
    min_max_normalize,
    entropy_weight,
    topsis,
    grey_relational_analysis,
    rank_desc
)

DATADIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------
# 1) Spearman秩相关（含p值）
# ------------------------------
def spearman_with_pvalue(rank_a: np.ndarray, rank_b: np.ndarray, seed: int = 20260119):
    """
    计算Spearman秩相关系数与p值。
    说明：
      - 若scipy可用，则使用scipy.stats.spearmanr（最标准）
      - 若scipy不可用，则使用：Spearman rho + Monte Carlo置换检验近似p值
        （n=10，置换检验足够稳定；同时固定随机种子保证可复现）
    """
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)

    try:
        from scipy.stats import spearmanr  # type: ignore
        rho, p = spearmanr(rank_a, rank_b)
        return float(rho), float(p), "scipy"
    except Exception:
        # 手工计算 rho（对秩向量计算皮尔逊相关）
        ra = rank_a
        rb = rank_b
        ra_c = ra - ra.mean()
        rb_c = rb - rb.mean()
        denom = (np.sqrt((ra_c ** 2).sum()) *
                 np.sqrt((rb_c ** 2).sum())) + 1e-12
        rho = float((ra_c * rb_c).sum() / denom)

        # Monte Carlo置换检验（双侧p值）
        rng = np.random.default_rng(seed)
        n_perm = 50000  # 可复现且足够稳定
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(rb)
            perm_c = perm - perm.mean()
            denom_p = (np.sqrt((ra_c ** 2).sum()) *
                       np.sqrt((perm_c ** 2).sum())) + 1e-12
            rho_p = float((ra_c * perm_c).sum() / denom_p)
            if abs(rho_p) >= abs(rho):
                count += 1
        p = (count + 1) / (n_perm + 1)
        return rho, float(p), "permutation"


# ------------------------------
# 2) 等级划分（按文档阈值）
# ------------------------------
def assign_grade_by_score(S: float) -> str:
    """
    文档等级划分：
      A+ : S >= 0.60
      A  : 0.30 <= S < 0.60
      B  : 0.25 <= S < 0.30
      C  : S < 0.25
    """
    if S >= 0.60:
        return "A+"
    if S >= 0.30:
        return "A"
    if S >= 0.25:
        return "B"
    return "C"


# ------------------------------
# 3) 六维度得分（基于X_norm的维度均值）
# ------------------------------
def compute_dimension_scores(X_norm: np.ndarray, indicator_names: list, countries: np.ndarray) -> pd.DataFrame:
    """
    根据文档九章定义：维度得分 = 该维度下归一化指标均值
    注意：这里使用X_norm（Min-Max后的x'），与文档一致。

    维度映射（与问题一/文档一致，24指标）：
      T: 人才
      A: 应用
      P: 政策
      R: 研发
      I: 基础设施
      O: 产出
    """
    idx = {name: j for j, name in enumerate(indicator_names)}

    # 这里采用你给的24指标命名（不带T1_前缀）
    groups = {
        "T_Talent": ["AI研究人员数量", "顶尖AI学者数量", "AI毕业生数量"],
        "A_Application": ["AI企业数量", "AI市场规模", "AI应用渗透率", "大模型数量"],
        "P_Policy": ["AI社会信任度", "AI政策数量", "AI补贴金额"],
        "R_RnD": ["企业研发支出", "政府AI投资", "国际AI投资"],
        "I_Infrastructure": ["5G覆盖率", "GPU集群规模", "互联网带宽", "互联网普及率", "电能生产", "AI算力平台", "数据中心数量", "TOP500上榜数"],
        "O_Output": ["AI_Book数量", "AI_Dataset数量", "GitHub项目数"],
    }

    # 计算每个维度的均值
    rows = []
    for i, country in enumerate(countries):
        row = {"Country": country}
        for dim, feats in groups.items():
            cols = []
            for f in feats:
                if f not in idx:
                    raise KeyError(f"维度映射需要的指标列缺失：{f}")
                cols.append(idx[f])
            row[dim] = float(np.mean(X_norm[i, cols]))
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------
# 4) 敏感性分析（逐指标扰动）
# ------------------------------
def sensitivity_analysis(
    X_norm: np.ndarray,
    base_weights: np.ndarray,
    countries: np.ndarray,
    deltas=None
):
    """
    按文档第八章执行：
      - 对每个指标权重 w_j 分别施加扰动：w_j*(1+delta)
      - delta ∈ {-30%, -15%, 0, +15%, +30%}
      - 扰动后全体权重归一化
      - 每次扰动重新计算TOPSIS排名
      - 统计每个国家在所有扰动下的排名极差 Range

    返回：
      rank_matrix_df: (n_countries, m*len(deltas)) 每列对应一次扰动的排名
      range_df: (n_countries,) 每个国家排名极差
    """
    if deltas is None:
        deltas = [-0.30, -0.15, 0.0, 0.15, 0.30]

    X_norm = np.asarray(X_norm, dtype=float)
    W = np.asarray(base_weights, dtype=float)

    n, m = X_norm.shape
    n_tests = m * len(deltas)

    # 保存每次扰动的排名
    rank_matrix = np.zeros((n, n_tests), dtype=int)
    col_names = []

    col_idx = 0
    for j in range(m):
        for d in deltas:
            Wp = W.copy()
            Wp[j] = W[j] * (1.0 + d)

            # 确保非负（极端情况下权重可能被扰动到负数，需截断）
            Wp = np.maximum(Wp, 0.0)

            # 归一化
            Wp = Wp / (Wp.sum() + 1e-12)

            # 重新TOPSIS
            C, _, _ = topsis(X_norm, Wp)
            ranks = rank_desc(C)

            rank_matrix[:, col_idx] = ranks
            col_names.append(f"ind{j+1:02d}_delta{d:+.2f}")
            col_idx += 1

    # 极差 Range
    ranges = rank_matrix.max(axis=1) - rank_matrix.min(axis=1)

    rank_matrix_df = pd.DataFrame(
        rank_matrix, index=countries, columns=col_names).reset_index()
    rank_matrix_df = rank_matrix_df.rename(columns={"index": "Country"})

    range_df = pd.DataFrame({
        "Country": countries,
        "Rank_Range": ranges
    })

    return rank_matrix_df, range_df


# ------------------------------
# 5) 主流程：严格对齐文档并输出全部结果
# ------------------------------
def run_task2_evaluation(
    data_path: str,
    output_dir: str = os.path.join(DATADIR, "outputs"),
    rho_grey: float = 0.5
):
    # 读取数据（与代码同目录）
    data = pd.read_csv(data_path, encoding="utf-8-sig")

    # 基本字段检查
    if "国家" not in data.columns:
        raise KeyError("数据文件缺少列：国家")

    countries = data["国家"].values
    indicators = [c for c in data.columns if c != "国家"]

    # 文档对齐：必须24指标
    if len(indicators) != 24:
        raise ValueError(f"指标数量不为24（当前={len(indicators)}），请确认使用问题一的24指标数据文件。")

    X = data[indicators].values

    # ------------------------------
    # 数据标准化（Min-Max）
    # ------------------------------
    X_norm = min_max_normalize(X)

    # ------------------------------
    # 模型1：熵权法
    # ------------------------------
    weights, entropy, redundancy = entropy_weight(X_norm)

    weight_df = pd.DataFrame({
        "Indicator": indicators,
        "Entropy": entropy,
        "Redundancy": redundancy,
        "Weight": weights
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    # ------------------------------
    # 模型2：TOPSIS
    # ------------------------------
    C, D_plus, D_minus = topsis(X_norm, weights)
    rank_topsis = rank_desc(C)

    topsis_df = pd.DataFrame({
        "Country": countries,
        "D_plus": D_plus,
        "D_minus": D_minus,
        "TOPSIS_Score": C,
        "TOPSIS_Rank": rank_topsis
    }).sort_values("TOPSIS_Rank").reset_index(drop=True)

    # ------------------------------
    # 模型3：灰色关联（加权）
    # ------------------------------
    gamma = grey_relational_analysis(X_norm, weights, rho=rho_grey)
    rank_grey = rank_desc(gamma)

    grey_df = pd.DataFrame({
        "Country": countries,
        "Grey_Relation": gamma,
        "Grey_Rank": rank_grey
    }).sort_values("Grey_Rank").reset_index(drop=True)

    # ------------------------------
    # 交叉验证：Spearman秩相关（含p值）
    # ------------------------------
    comp = topsis_df[["Country", "TOPSIS_Rank"]].merge(
        grey_df[["Country", "Grey_Rank"]],
        on="Country", how="inner"
    )

    spearman_rho, spearman_p, spearman_method = spearman_with_pvalue(
        comp["TOPSIS_Rank"].values,
        comp["Grey_Rank"].values
    )

    spearman_df = pd.DataFrame([{
        "Spearman_rho": spearman_rho,
        "p_value": spearman_p,
        "method": spearman_method
    }])

    # ------------------------------
    # 综合评估：Borda + 综合得分
    # ------------------------------
    final_df = topsis_df.merge(
        grey_df, on="Country", how="inner"
    )

    # Borda得分（名次之和越小越好）
    final_df["Borda"] = final_df["TOPSIS_Rank"] + final_df["Grey_Rank"]

    # 综合得分（文档：alpha=beta=0.5）
    final_df["Comprehensive_Score"] = 0.5 * \
        final_df["TOPSIS_Score"] + 0.5 * final_df["Grey_Relation"]

    # 最终排名（按Borda排序；若Borda相同则按综合得分降序作为稳定tie-break）
    final_df = final_df.sort_values(["Borda", "Comprehensive_Score"], ascending=[
                                    True, False]).reset_index(drop=True)
    final_df["Final_Rank"] = np.arange(1, len(final_df) + 1)

    # 等级划分（按综合得分阈值）
    final_df["Grade"] = final_df["Comprehensive_Score"].apply(
        assign_grade_by_score)

    # ------------------------------
    # 敏感性分析（文档第八章）
    # ------------------------------
    rank_matrix_df, range_df = sensitivity_analysis(X_norm, weights, countries)

    # ------------------------------
    # 六维度得分（文档第九章）
    # ------------------------------
    dim_df = compute_dimension_scores(X_norm, indicators, countries)

    # ------------------------------
    # 输出全部结果
    # ------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # 与文档表结构相匹配的输出（英文列名，便于代码稳健；论文可再映射中文表头）
    weight_df.to_csv(os.path.join(output_dir, "weights_entropy.csv"),
                     index=False, encoding="utf-8-sig")
    topsis_df.to_csv(os.path.join(output_dir, "result_topsis.csv"),
                     index=False, encoding="utf-8-sig")
    grey_df.to_csv(os.path.join(
        output_dir, "result_grey_relation.csv"), index=False, encoding="utf-8-sig")
    comp.to_csv(os.path.join(output_dir, "result_comparison.csv"),
                index=False, encoding="utf-8-sig")
    final_df.to_csv(os.path.join(
        output_dir, "result_final_ranking.csv"), index=False, encoding="utf-8-sig")

    spearman_df.to_csv(os.path.join(
        output_dir, "result_spearman.csv"), index=False, encoding="utf-8-sig")
    rank_matrix_df.to_csv(os.path.join(
        output_dir, "result_sensitivity_rank_matrix.csv"), index=False, encoding="utf-8-sig")
    range_df.to_csv(os.path.join(
        output_dir, "result_sensitivity_range.csv"), index=False, encoding="utf-8-sig")
    dim_df.to_csv(os.path.join(
        output_dir, "result_dimension_scores.csv"), index=False, encoding="utf-8-sig")

    return {
        "weights": weight_df,
        "topsis": topsis_df,
        "grey": grey_df,
        "comparison": comp,
        "final": final_df,
        "spearman": spearman_df,
        "sensitivity_rank_matrix": rank_matrix_df,
        "sensitivity_range": range_df,
        "dimension_scores": dim_df,
    }
