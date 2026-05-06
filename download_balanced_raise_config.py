#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接在脚本里修改配置，不需要命令行参数。

新增功能：
1. 多线程并发下载
2. 总进度条
3. 实时下载速度统计
4. 已下载容量统计
5. 成功/失败计数

使用方法：
1. 打开脚本
2. 修改 main() 里的这几个变量：
   - csv_path
   - keywords
   - total
   - output_dir
   - file_format
   - num_workers
3. 直接运行：
   python download_balanced_raise_config.py
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from tqdm import tqdm


def normalize_token(token: str) -> str:
    return token.strip().lower()


def split_keywords(value) -> List[str]:
    if pd.isna(value):
        return []
    return [normalize_token(x) for x in str(value).split(";") if x.strip()]


def discover_all_keywords(df: pd.DataFrame) -> List[str]:
    s = set()
    for value in df["Keywords"].fillna(""):
        for kw in split_keywords(value):
            s.add(kw)
    return sorted(s)


def filter_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    keyword = normalize_token(keyword)
    mask = df["Keywords"].fillna("").apply(
        lambda x: keyword in split_keywords(x)
    )
    return df[mask].copy()


def infer_scene_bucket(row: pd.Series, target_keyword: str) -> str:
    scene_mode = str(row.get("Scene Mode", "")).strip()
    if scene_mode and scene_mode.lower() != "nan":
        return f"scene_mode::{scene_mode.lower()}"

    kws = split_keywords(row.get("Keywords", ""))
    others = sorted([k for k in kws if k != target_keyword])
    if others:
        return "co_tags::" + "|".join(others)

    return "unknown"


def balanced_take_from_groups(groups: Dict[str, List[int]], n: int, rng: random.Random) -> List[int]:
    pools = {g: items[:] for g, items in groups.items()}
    for items in pools.values():
        rng.shuffle(items)

    selected = []
    active_groups = [g for g, items in pools.items() if items]

    while len(selected) < n and active_groups:
        rng.shuffle(active_groups)
        next_active = []
        for g in active_groups:
            if len(selected) >= n:
                break
            if pools[g]:
                selected.append(pools[g].pop())
            if pools[g]:
                next_active.append(g)
        active_groups = next_active

    return selected


def allocate_counts_evenly(total: int, names: List[str]) -> Dict[str, int]:
    if not names:
        return {}
    base = total // len(names)
    remainder = total % len(names)
    result = {}
    for i, name in enumerate(names):
        result[name] = base + (1 if i < remainder else 0)
    return result


def sample_balanced(df: pd.DataFrame, keywords: List[str], total: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)

    available = df[(df["TIFF"].notna()) | (df["NEF"].notna())].copy()

    per_kw_df = {}
    for kw in keywords:
        sub = filter_by_keyword(available, kw)
        if len(sub) > 0:
            per_kw_df[kw] = sub

    if not per_kw_df:
        raise ValueError("没有找到任何匹配关键词且带下载链接的样本。")

    valid_keywords = list(per_kw_df.keys())
    target_counts = allocate_counts_evenly(total, valid_keywords)

    selected_indices = []
    used_global = set()

    for kw in valid_keywords:
        sub = per_kw_df[kw]
        sub = sub[~sub.index.isin(used_global)].copy()
        if len(sub) == 0:
            continue

        sub["__bucket__"] = sub.apply(lambda row: infer_scene_bucket(row, kw), axis=1)
        groups = defaultdict(list)
        for idx, row in sub.iterrows():
            groups[row["__bucket__"]].append(idx)

        want = min(target_counts[kw], len(sub))
        picked = balanced_take_from_groups(groups, want, rng)

        selected_indices.extend(picked)
        used_global.update(picked)

    if len(selected_indices) < total:
        remaining = available[~available.index.isin(used_global)].copy()
        remaining_idx = remaining.index.tolist()
        rng.shuffle(remaining_idx)
        need = total - len(selected_indices)
        selected_indices.extend(remaining_idx[:need])

    sampled = available.loc[selected_indices].copy()

    if len(sampled) > total:
        sampled = sampled.sample(n=total, random_state=seed).copy()

    return sampled


def get_url_and_ext(row: pd.Series, fmt: str):
    if fmt == "tiff":
        url = str(row.get("TIFF", "")).strip()
        ext = ".TIF"
    else:
        url = str(row.get("NEF", "")).strip()
        ext = ".NEF"

    if not url or url.lower() == "nan":
        return None, None

    return url, ext


def format_size(num_bytes: float) -> str:
    mb = num_bytes / 1024 / 1024
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024
    return f"{gb:.2f} GB"


class DownloadStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_bytes = 0
        self.success = 0
        self.fail = 0
        self.start_time = time.time()

    def add_success(self, nbytes: int):
        with self.lock:
            self.success += 1
            self.total_bytes += nbytes

    def add_fail(self):
        with self.lock:
            self.fail += 1

    def snapshot(self):
        with self.lock:
            elapsed = max(time.time() - self.start_time, 1e-6)
            speed = self.total_bytes / elapsed
            return self.success, self.fail, self.total_bytes, speed


def download_file(
    url: str,
    out_path: Path,
    timeout: int = 30,
    max_retries: int = 2,
    chunk_size: int = 1024 * 1024,
) -> Tuple[bool, int, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proxies = {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897",
    }

    for attempt in range(max_retries + 1):
        try:
            downloaded_bytes = 0
            with requests.get(url, stream=True, timeout=timeout, proxies=proxies) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
            return True, downloaded_bytes, ""
        except Exception as e:
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            if attempt == max_retries:
                return False, 0, str(e)

    return False, 0, "unknown error"


def download_one(row, file_format, output_dir, timeout, max_retries):
    url, ext = get_url_and_ext(row, file_format)
    if not url:
        return {
            "ok": False,
            "row": row,
            "bytes": 0,
            "reason": f"missing {file_format} url",
            "saved_path": "",
            "filename": "",
        }

    file_stem = str(row["File"]).strip()
    filename = file_stem + ext
    out_path = output_dir / filename

    ok, nbytes, reason = download_file(
        url=url,
        out_path=out_path,
        timeout=timeout,
        max_retries=max_retries,
    )

    return {
        "ok": ok,
        "row": row,
        "bytes": nbytes,
        "reason": reason,
        "saved_path": str(out_path) if ok else "",
        "filename": filename,
    }


def main():
    # ========= 直接改这里 =========
    csv_path = r"G:\project\LAMALocal\Dataset\RAISE_255.csv"
    keywords = ["outdoor", "indoor", "people", "buildings"]
    total = 100
    output_dir = r"./Dataset/Origin"
    file_format = "tiff"   # 可选: "tiff" 或 "nef"
    seed = 42
    timeout = 30
    max_retries = 2
    num_workers = 8        # 并发线程数，网络好可以调大到 12/16
    # =============================

    csv_path = Path(csv_path)

    # 相对路径时，以脚本所在目录为基准
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {"File", "Keywords", "TIFF", "NEF"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    if not keywords:
        keywords = discover_all_keywords(df)

    keywords = [normalize_token(x) for x in keywords if str(x).strip()]
    if not keywords:
        raise ValueError("没有可用关键词。")

    sampled = sample_balanced(df, keywords, total, seed)

    sampled["matched_keywords"] = sampled["Keywords"].fillna("").apply(
        lambda x: "; ".join([kw for kw in keywords if kw in split_keywords(x)])
    )

    rows = [row for _, row in sampled.iterrows()]
    stats = DownloadStats()

    success_records = []
    fail_records = []

    print("开始下载")
    print(f"目标数量: {len(rows)}")
    print(f"输出目录: {output_dir}")
    print(f"下载格式: {file_format}")
    print(f"线程数: {num_workers}")
    print("")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                download_one,
                row,
                file_format,
                output_dir,
                timeout,
                max_retries,
            )
            for row in rows
        ]

        with tqdm(total=len(futures), desc="总进度", ncols=120) as pbar:
            for future in as_completed(futures):
                result = future.result()

                if result["ok"]:
                    stats.add_success(result["bytes"])
                    rec = result["row"].to_dict()
                    rec["saved_path"] = result["saved_path"]
                    rec["download_bytes"] = result["bytes"]
                    success_records.append(rec)
                else:
                    stats.add_fail()
                    fail_records.append({
                        **result["row"].to_dict(),
                        "reason": result["reason"],
                    })

                success_cnt, fail_cnt, total_bytes, speed = stats.snapshot()

                pbar.update(1)
                pbar.set_postfix({
                    "success": success_cnt,
                    "fail": fail_cnt,
                    "downloaded": format_size(total_bytes),
                    "speed": f"{speed / 1024 / 1024:.2f} MB/s",
                })

    success_df = pd.DataFrame(success_records)
    fail_df = pd.DataFrame(fail_records)

    success_csv = output_dir / "downloaded_metadata.csv"
    fail_csv = output_dir / "failed_metadata.csv"

    success_df.to_csv(success_csv, index=False, encoding="utf-8-sig")
    fail_df.to_csv(fail_csv, index=False, encoding="utf-8-sig")

    success_cnt, fail_cnt, total_bytes, speed = stats.snapshot()

    print("\n========== 完成 ==========")
    print(f"目标数量: {total}")
    print(f"实际成功下载: {success_cnt}")
    print(f"下载失败: {fail_cnt}")
    print(f"累计下载大小: {format_size(total_bytes)}")
    print(f"平均下载速度: {speed / 1024 / 1024:.2f} MB/s")
    print(f"成功清单: {success_csv}")
    print(f"失败清单: {fail_csv}")


if __name__ == "__main__":
    main()