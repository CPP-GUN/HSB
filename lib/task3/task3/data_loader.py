# -*- coding: utf-8 -*-
"""
data_loader.py
负责从 DATA/ 目录递归读取 24 个指标 CSV，并构建统一的长表面板数据：
(year, country, indicator, value)

要求：
- 每个 CSV 格式类似：
  年份,美国,中国,英国,德国,韩国,日本,法国,加拿大,阿联酋,印度
  2016, ...
  ...
  2025, ...

- DATA 目录结构可为多层子文件夹（T/A/P/R/I/O 等），此模块自动递归查找匹配文件。
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import os

DATADIR = os.path.dirname(os.path.abspath(__file__))

# 固定国家顺序（与题目/数据一致）
COUNTRIES_CN: List[str] = ["美国", "中国", "英国",
                           "德国", "韩国", "日本", "法国", "加拿大", "阿联酋", "印度"]


@dataclass(frozen=True)
class IndicatorSpec:
    """指标规范：中文指标名 + 可能的文件名关键词（用于匹配）"""
    name: str
    keywords: Tuple[str, ...]


def default_indicator_specs() -> List[IndicatorSpec]:
    """
    统一问题1/2/3的 24 指标，并提供文件匹配关键词。
    说明：
    - 你的文件名通常就等于指标名（或非常接近），这里做了容错关键词。
    - 关键词越具体越好，避免误匹配。
    """
    return [
        IndicatorSpec("AI研究人员数量", ("AI研究人员数量",)),
        IndicatorSpec("顶尖AI学者数量", ("顶尖AI学者数量",)),
        IndicatorSpec("AI毕业生数量", ("AI毕业生数量",)),
        IndicatorSpec("AI企业数量", ("AI企业数量",)),
        IndicatorSpec("AI市场规模", ("AI市场规模", "市场规模")),
        IndicatorSpec("AI应用渗透率", ("AI应用渗透率", "渗透率")),
        IndicatorSpec("大模型数量", ("大模型数量",)),
        IndicatorSpec("AI社会信任度", ("AI社会信任度", "社会信任")),
        IndicatorSpec("AI政策数量", ("AI政策数量", "政策数量")),
        IndicatorSpec("AI补贴金额", ("AI补贴金额", "补贴")),
        IndicatorSpec("企业研发支出", ("企业研发支出", "研发支出")),
        IndicatorSpec("政府AI投资", ("政府AI投资", "政府AI投入", "政府投资")),
        IndicatorSpec("5G覆盖率", ("5G覆盖率", "5G覆盖")),
        IndicatorSpec("GPU集群规模", ("GPU集群规模", "GPU")),
        IndicatorSpec("互联网带宽", ("互联网带宽", "带宽")),
        IndicatorSpec("互联网普及率", ("互联网普及率", "普及率")),
        IndicatorSpec("电能生产", ("电能生产", "TWh")),
        IndicatorSpec("AI算力平台", ("AI算力平台", "算力平台")),
        IndicatorSpec("数据中心数量", ("数据中心数量", "数据中心")),
        IndicatorSpec("TOP500上榜数", ("TOP500上榜数", "TOP500")),
        IndicatorSpec("AI_Book数量", ("AI_Book数量", "AI_Book")),
        IndicatorSpec("AI_Dataset数量", ("AI_Dataset数量", "AI_Dataset")),
        IndicatorSpec("GitHub项目数", ("GitHub项目数", "GitHub")),
        IndicatorSpec("国际AI投资", ("国际AI投资", "国际AI投入", "国际投资")),
    ]


def _normalize_filename(s: str) -> str:
    """将文件名做轻度规范化，便于关键词匹配。"""
    s = s.strip().lower()
    # 去掉常见括号内容与空白
    for ch in [" ", "（", "）", "(", ")", "—", "-", "_"]:
        s = s.replace(ch, "")
    return s


def find_indicator_files(data_dir: Path, specs: List[IndicatorSpec]) -> Dict[str, Path]:
    """
    在 DATA/ 下递归查找每个指标对应的 CSV 文件路径。
    匹配策略：
    - 对每个指标 spec，寻找文件名中包含其任一关键词的 CSV。
    - 若出现多个候选：优先选择“匹配关键词最长”的那个；仍冲突则报错提示用户处理。
    """
    all_csv = list(data_dir.rglob("*.csv"))
    if not all_csv:
        raise FileNotFoundError(f"未在目录 {data_dir} 下找到任何 CSV 文件。")

    # 预处理：文件名规范化
    file_norm_map = {p: _normalize_filename(p.stem) for p in all_csv}

    result: Dict[str, Path] = {}
    for spec in specs:
        candidates: List[Tuple[int, Path]] = []
        for p, norm_name in file_norm_map.items():
            for kw in spec.keywords:
                kw_norm = _normalize_filename(kw)
                if kw_norm and kw_norm in norm_name:
                    candidates.append((len(kw_norm), p))
        if not candidates:
            raise FileNotFoundError(
                f"指标【{spec.name}】未找到匹配的 CSV 文件。请检查 DATA 目录下文件命名。"
            )

        # 取匹配最长关键词的候选
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_len = candidates[0][0]
        best = [p for l, p in candidates if l == best_len]

        if len(best) > 1:
            # 若多个同分候选，通常说明关键词过短或文件重名
            msg = "\n".join(str(x) for x in best)
            raise RuntimeError(
                f"指标【{spec.name}】匹配到多个候选文件，无法自动决定：\n{msg}\n"
                f"建议：将文件名改得更具体（包含完整指标名），或在 keywords 中增加更精确关键词。"
            )

        result[spec.name] = best[0]

    return result


def read_indicator_csv(file_path: Path, indicator_name: str, countries: List[str]) -> pd.DataFrame:
    """
    读取单个指标 CSV，输出长表：
    year, country, indicator, value
    """
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if "年份" not in df.columns:
        raise ValueError(f"{file_path} 缺少列【年份】。")

    missing_c = [c for c in countries if c not in df.columns]
    if missing_c:
        raise ValueError(f"{file_path} 缺少国家列：{missing_c}")

    # 仅保留年份 + 国家列
    df = df[["年份"] + countries].copy()
    df.rename(columns={"年份": "year"}, inplace=True)

    # 宽转长
    long_df = df.melt(id_vars=["year"], var_name="country", value_name="value")
    long_df["indicator"] = indicator_name

    # 类型清洗
    long_df["year"] = pd.to_numeric(
        long_df["year"], errors="raise").astype(int)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    # 这里不做缺失处理（你已说明数据完整）；若未来有缺失，可在预测模块兜底处理
    return long_df


def load_all_data(
    base_dir: Path,
    specs: Optional[List[IndicatorSpec]] = None,
    countries: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    主函数：读取所有指标，构建统一面板长表。
    返回：
    - panel_long: columns = [year, country, indicator, value]
    - indicator_files: 指标 -> 文件路径映射
    """
    specs = specs or default_indicator_specs()
    countries = countries or COUNTRIES_CN

    data_dir = base_dir / "DATA"
    if not data_dir.exists():
        raise FileNotFoundError(f"未找到 DATA 目录：{data_dir}")

    indicator_files = find_indicator_files(data_dir, specs)

    frames = []
    for ind_name, path in indicator_files.items():
        frames.append(read_indicator_csv(path, ind_name, countries))

    panel_long = pd.concat(frames, ignore_index=True)

    # 排序便于复现与查看
    panel_long.sort_values(["indicator", "country", "year"], inplace=True)
    panel_long.reset_index(drop=True, inplace=True)

    return panel_long, indicator_files
