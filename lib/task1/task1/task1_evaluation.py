"""
task1_evaluation.py
-------------------------------------------------
用途：
实现建模文档第 7 节：
- 综合得分 S_i = sum(w_k * PC_ik)
- 归一化到 [0,1]
- 国家排名

这是 Task 1 的最终量化输出。
-------------------------------------------------
"""

import os
from task1_utils import load_data, run_pca, select_m, comprehensive_score

os.makedirs("outputs", exist_ok=True)


def main():
    countries, indicators, X = load_data()
    Z, variance_df, loadings = run_pca(X, indicators)
    m = select_m(variance_df)

    result = comprehensive_score(Z, variance_df, m, countries)
    result.to_csv("outputs/comprehensive_evaluation.csv", index=False)

    print("Comprehensive evaluation completed.")
    print(result)


if __name__ == "__main__":
    main()
