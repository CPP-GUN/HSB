import pandas as pd
import numpy as np
import os

DATADIR = os.path.dirname(os.path.abspath(__file__))

def min_max_normalize_numeric(df, id_cols):
    """
    仅对数值列进行 Min-Max 归一化
    自动处理字符串、缺失值
    """
    df_norm = df.copy()

    # 识别指标列（非 ID 列）
    indicator_cols = [c for c in df.columns if c not in id_cols]

    # 强制转为数值，非法字符 → NaN
    for col in indicator_cols:
        df_norm[col] = pd.to_numeric(df_norm[col], errors="coerce")

    # 缺失值处理：用该指标的均值填补
    for col in indicator_cols:
        if df_norm[col].isna().any():
            mean_val = df_norm[col].mean()
            df_norm[col] = df_norm[col].fillna(mean_val)

    # Min-Max 标准化
    for col in indicator_cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()

        if max_val == min_val:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

    return df_norm


def main():
    input_file = os.path.join(DATADIR, "data_raw_indicators.csv")
    output_file = os.path.join(DATADIR, "data_standardized.csv")

    df_raw = pd.read_csv(input_file)

    # 国家列（只保留，不参与计算）
    id_cols = ["国家"]

    df_std = min_max_normalize_numeric(df_raw, id_cols)

    df_std.to_csv(output_file, index=False)

    print("✔ 标准化完成：data_standardized.csv")
    print("✔ 指标范围检查（应在 [0,1]）：")
    print(df_std.drop(columns=id_cols).describe().loc[["min", "max"]])


if __name__ == "__main__":
    main()
