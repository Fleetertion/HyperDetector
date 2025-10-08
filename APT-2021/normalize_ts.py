#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from dateutil.parser import parse as _parse
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser(
        description="把混合格式的时间字符串全自动解析为 pandas.Timestamp"
    )
    ap.add_argument("csv", type=Path, help="原始 APT2021 CSV 文件路径")
    ap.add_argument("--out_csv", type=Path, default="with_timestamp.csv",
                    help="带新 Timestamp 列的输出 CSV")
    args = ap.parse_args()

    # 1) 全当字符串读进来，防止 pandas 预先尝试解析
    df = pd.read_csv(args.csv, dtype=str)

    # 2) 找到第一个包含 “time” 的列
    ts_col = next((c for c in df.columns if "time" in c.lower()), None)
    if ts_col is None:
        raise RuntimeError("找不到任何包含 'time' 的列！")

    # 3) 逐行用 dateutil 解析
    def safe_parse(s):
        try:
            return _parse(s)
        except Exception as e:
            # 如果某行实在跑不通，你可以在这里打印一下 s，再决定跳过还是报错
            raise ValueError(f"无法解析时间字符串 {s!r}: {e}")

    tqdm.pandas(desc="Parsing timestamps")
    df["Timestamp"] = df[ts_col].progress_apply(safe_parse)

    # 4) 保存到新的 CSV（也可以直接继续后续处理）
    df.to_csv(args.out_csv, index=False)
    print(f"已生成带统一 `Timestamp` 列的新文件：{args.out_csv}")

if __name__ == "__main__":
    main()
