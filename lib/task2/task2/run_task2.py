# run_task2.py
# =====================================================
# 问题二：一键可复现运行入口（不涉及可视化）
# 用法：python run_task2.py
# 输出：同目录 outputs/ 下生成所有结果表
# =====================================================

from evaluation_pipeline import run_task2_evaluation
import os

DATADIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    results = run_task2_evaluation(
        data_path=os.path.join(DATADIR, "data_raw_indicators.csv"),
        output_dir=os.path.join(DATADIR, "outputs"),
        rho_grey=0.5
    )

    final_df = results["final"]
    spearman_df = results["spearman"]
    range_df = results["sensitivity_range"]

    print("=" * 70)
    print("问题二运行完成（严格对齐：熵权法 + TOPSIS + 灰色关联 + Spearman + Borda + 等级 + 敏感性 + 六维度）")
    print("=" * 70)

    # Spearman一致性检验（文档要求）
    rho = spearman_df.loc[0, "Spearman_rho"]
    p = spearman_df.loc[0, "p_value"]
    method = spearman_df.loc[0, "method"]
    print(f"\nSpearman秩相关检验：rho = {rho:.4f}, p = {p:.4f} (method={method})")

    # 打印最终排名核心列
    print("\n2025 AI竞争力最终排名（Top 10）：")
    print(final_df[[
        "Final_Rank", "Country",
        "Comprehensive_Score", "Grade",
        "TOPSIS_Rank", "Grey_Rank", "Borda"
    ]])

    # 打印敏感性Range摘要（文档要求的稳定性指标）
    print("\n敏感性分析：各国排名极差（Range）摘要：")
    print(range_df.sort_values("Rank_Range").reset_index(drop=True))

    print("\n所有结果已写入：outputs/ 目录")
