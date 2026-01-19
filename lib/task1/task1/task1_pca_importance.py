"""
task1_pca_importance.py
-------------------------------------------------
用途：
实现建模文档第 5 节：
- PCA 分析
- 主成分选择（累计解释率 ≥ 0.85）
- 要素重要性计算 I_j

输出 PCA 方差、载荷与重要性结果。
-------------------------------------------------
"""

import os
from task1_utils import load_data, run_pca, select_m, factor_importance

DATADIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(DATADIR, "outputs"), exist_ok=True)


def main():
    countries, indicators, X = load_data()

    Z, variance_df, loadings = run_pca(X, indicators)
    m = select_m(variance_df)

    variance_df.to_csv(os.path.join(DATADIR, "outputs", "pca_variance.csv"))
    loadings.to_csv(os.path.join(DATADIR, "outputs", "pca_loadings.csv"))

    importance = factor_importance(loadings, variance_df, m)
    importance.to_csv(os.path.join(DATADIR, "outputs", "factor_importance.csv"))

    print("PCA & factor importance completed.")
    print(f"Selected principal components: {m}")


if __name__ == "__main__":
    main()
