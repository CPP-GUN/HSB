import pandas as pd
import numpy as np

# 读取PCA载荷矩阵
pca_df = pd.read_csv('pca_loadings.csv', index_col=0)
print("=== PCA载荷矩阵 ===")
print(pca_df)

# PCA方差贡献率（真实数据）
variance_ratios = [0.5898, 0.1859, 0.1016]

# 计算要素重要性
importance = {}
for idx, indicator in enumerate(pca_df.index):
    # I_j = Σ(loading²_jk × variance_ratio_k)
    score = 0
    for i, col in enumerate(['PC1', 'PC2', 'PC3']):
        loading = pca_df.loc[indicator, col]
        score += (loading ** 2) * variance_ratios[i]
    importance[indicator] = score

# 排序
importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n=== Top 10要素重要性排名（真实数据）===")
for rank, (indicator, score) in enumerate(importance_sorted[:10], 1):
    print(f"{rank}. {indicator}: I={score:.4f} ({score/importance_sorted[0][1]*100:.1f}%)")

print("\n=== Top 5要素（用于文档）===")
for rank, (indicator, score) in enumerate(importance_sorted[:5], 1):
    print(f"{rank}. {indicator} ($I={score:.3f}$)")

# 读取相关性矩阵
corr = pd.read_csv('correlation_matrix.csv', index_col=0)
print("\n=== 最强相关对（|r| > 0.7）===")

strong_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        r_val = corr.iloc[i, j]
        if pd.notna(r_val) and abs(r_val) > 0.7:
            strong_corr.append((corr.columns[i], corr.columns[j], r_val))

# 按相关系数绝对值排序
strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\n共找到{len(strong_corr)}对强相关关系")
print("\n前15对最强相关：")
for rank, (ind1, ind2, r) in enumerate(strong_corr[:15], 1):
    print(f"{rank}. {ind1} ↔ {ind2}: r={r:.3f}")

# 检查文档中提到的相关系数
print("\n=== 验证文档中的相关系数 ===")
targets = [
    ("AI研究人员数量", "AI_Book数量", "人才→创新"),
    ("企业研发支出", "AI研究人员数量", "研发→人才"),
    ("企业研发支出", "AI_Book数量", "研发→创新"),
    ("企业研发支出", "AI市场规模", "研发→产业"),
    ("GPU集群规模", "AI市场规模", "基础设施→产业"),
    ("AI政策数量", "企业研发支出", "政策→研发"),
]

for ind1, ind2, label in targets:
    if ind1 in corr.index and ind2 in corr.columns:
        r = corr.loc[ind1, ind2]
        if pd.notna(r):
            print(f"{label}: {ind1} ↔ {ind2} = r={r:.3f}")
        else:
            print(f"{label}: {ind1} ↔ {ind2} = 缺失数据")
    else:
        print(f"{label}: 指标不存在")
