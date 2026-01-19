"""
mainpip.py
-------------------------------------------------
用途：
Task 1（问题一）统一主入口脚本。

按建模文档流程顺序依次执行：
1. 要素关联分析（Pearson 相关性 + 强相关对 + 层次聚类）
2. PCA 分析与要素重要性评估
3. 基于 PCA 的综合评估与国家排名

本脚本不包含任何模型细节，仅负责流程调度，
确保结果可复现、结构清晰、便于整体运行。
-------------------------------------------------
"""

import task1_correlation_cluster
import task1_pca_importance
import task1_evaluation


def main():
    print("=== Task 1: AI Development Capability Analysis ===")

    print("\n[Step 1] Correlation & Hierarchical Clustering")
    task1_correlation_cluster.main()

    print("\n[Step 2] PCA & Factor Importance")
    task1_pca_importance.main()

    print("\n[Step 3] Comprehensive Evaluation & Ranking")
    task1_evaluation.main()

    print("\n=== Task 1 Finished Successfully ===")


if __name__ == "__main__":
    main()
