"""
task1_correlation_cluster.py
-------------------------------------------------
用途：
实现建模文档第 4 节：
- Pearson 相关系数矩阵 R
- 强相关对统计（|r|>0.7）
- 层次聚类（d=1-|r|，Average Linkage）

输出全部为后续分析和可视化所需的中间结果。
-------------------------------------------------
"""

import os
import pandas as pd
from task1_utils import load_data, pearson_correlation, strong_correlations, hierarchical_clustering


os.makedirs("outputs", exist_ok=True)


def main():
    countries, indicators, X = load_data()

    R = pearson_correlation(X, indicators)
    R.to_csv("outputs/correlation_matrix.csv")

    strong = strong_correlations(R)
    strong.to_csv("outputs/strong_correlations.csv", index=False)

    D, Z = hierarchical_clustering(R)
    D.to_csv("outputs/cluster_distance_matrix.csv")

    pd.DataFrame(Z).to_csv("outputs/cluster_linkage.csv", index=False)

    print("Correlation & clustering completed.")
    print(f"Strong correlation pairs: {len(strong)}")


if __name__ == "__main__":
    import pandas as pd
    main()
