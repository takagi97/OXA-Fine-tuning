#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read two jsonl data sources, filter and construct training dataset as required, and output as a single JSON file (array):
- File 1: PPL is first capped at 5, then select global top K by high to low; max X items per query; loss_type=0
- File 2: Select global top M by original PPL low to high; max X items per query; loss_type=1
- Output fields: instruction（query+SUFFIX）、input=""、output=response、history=[]、loss_type
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable

# =========================
# Hyperparameters
# =========================
FILE1_PATH = "AceReason-1.1-Math.uuid.gt.verify.true.shuf.DeepSeek-R1_Response_Qwen2.5-Math-7B_ppl_list.sample_gauss_mean3.0_std0.25_len.5w.1query1response.jsonl"  # Path to the first file
FILE2_PATH = "AceReason-1.1-Math.uuid.gt.verify.true.dps_7b_sample_list.dps_7b_sample_xverifier.false.dps_7b_sample_Qwen2.5-Math-7B_ppl_list_all.add_think.rm_request_fail.rm_len_25k.lowest_50000.1query1response.jsonl"  # Path to the second file
OUT_PATH   = "merged_7b_ce5w_LEN_ul5w_rm_len25k_1query1response.alpaca.json"  # Output JSONL file path

K = 50000   # Total global samples selected from File 1 (by high PPL, capped at 5)
M = 50000   # Total global samples selected from File 2 (by low PPL)
X = 1      # Max responses selected per query

# Template
SUFFIX = (
    "\nLet's reason step by step. Enclose the reasoning process within <think>...</think>, "
    "then summarize it and present the final answer within \\boxed{} — for example: "
    "<think>reasoning process here</think> \\boxed{answer here}."
)

# Field names
F1_QUERY_KEY = "query"
F1_RESP_LIST_KEY = "response"
F1_PPL_LIST_KEY = "ppl"

F2_QUERY_KEY = "query"
F2_RESP_LIST_KEY = "dps_7b_sample_list"
F2_PPL_LIST_KEY = "dps_7b_sample_Qwen2.5-Math-7B_ppl_list"

# =========================
# Data Structures
# =========================
@dataclass
class Pair:
    source: int               # 1 or 2: Source file origin
    query: str
    response: str
    ppl: float
    ppl_capped: float         # Used for File 1 (min(ppl, 5)), same as ppl for File 2
    loss_type: int            # File 1 = 0, File 2 = 1

# =========================
# Utility Functions
# =========================
def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_json_array(path: str, records: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def flatten_records(records: Iterable[Dict[str, Any]],
                    query_key: str,
                    resp_key: str,
                    ppl_key: str,
                    src_id: int,
                    cap_to_5: bool) -> List[Pair]:
    out: List[Pair] = []
    for item in records:
        query = item.get(query_key, "")
        resp_list = item.get(resp_key, [])
        ppl_list = item.get(ppl_key, [])

        if not isinstance(resp_list, list) or not isinstance(ppl_list, list):
            resp_list = [resp_list]
            ppl_list = [ppl_list]
            # continue
        n = min(len(resp_list), len(ppl_list))
        for i in range(n):
            resp = resp_list[i]
            ppl = ppl_list[i]
            # Skip outliers/invalid values
            try:
                ppl_f = float(ppl)
            except (TypeError, ValueError):
                continue
            if not isinstance(resp, str):
                continue

            ppl_capped = min(ppl_f, 5.0) if cap_to_5 else ppl_f
            out.append(Pair(
                source=src_id,
                query=str(query),
                response=resp,
                ppl=ppl_f,
                ppl_capped=ppl_capped,
                loss_type=0 if src_id == 1 else 1
            ))
    return out

def select_with_per_query_cap(
    pairs: List[Pair],
    key_fn,
    total_limit: int,
    per_query_cap: int,
    reverse: bool
) -> List[Pair]:
    """
    General selector:
    - Sort by key_fn(p) (reverse=True for descending, e.g., "highest priority")
    - Global max total_limit items
    - Max per_query_cap items per query
    """
    sorted_pairs = sorted(pairs, key=key_fn, reverse=reverse)
    kept: List[Pair] = []
    perq_count: Dict[str, int] = {}

    for p in sorted_pairs:
        if len(kept) >= total_limit:
            break
        cnt = perq_count.get(p.query, 0)
        if cnt >= per_query_cap:
            continue
        kept.append(p)
        perq_count[p.query] = cnt + 1

    return kept

def build_output_records(selected: List[Pair]) -> List[Dict[str, Any]]:
    out = []
    for p in selected:
        out.append({
            "instruction": f"{p.query}{SUFFIX}",
            "input": "",
            "output": p.response,
            "history": [],
            "loss_type": p.loss_type
        })
    return out

# =========================
# Main Flow
# =========================
def main():
    # Read and flatten both files
    file1_pairs = flatten_records(
        read_jsonl(FILE1_PATH),
        query_key=F1_QUERY_KEY,
        resp_key=F1_RESP_LIST_KEY,
        ppl_key=F1_PPL_LIST_KEY,
        src_id=1,
        cap_to_5=True,   # File 1: Max PPL value is 5 (upper truncation on PPL)
    )

    file2_pairs = flatten_records(
        read_jsonl(FILE2_PATH),
        query_key=F2_QUERY_KEY,
        resp_key=F2_RESP_LIST_KEY,
        ppl_key=F2_PPL_LIST_KEY,
        src_id=2,
        cap_to_5=False,  # File 2: No truncation, take lowest based on raw PPL
    )

    # Selection phase
    # File 1: By ppl_capped high to low, max K items; max X items per query
    sel1 = select_with_per_query_cap(
        file1_pairs,
        key_fn=lambda p: p.ppl_capped,
        total_limit=K,
        per_query_cap=X,
        reverse=True,  # High priority
    )

    # File 2: By ppl low to high, max M items; max X items per query
    sel2 = select_with_per_query_cap(
        file2_pairs,
        key_fn=lambda p: p.ppl,
        total_limit=M,
        per_query_cap=X,
        reverse=False,  # Low priority
    )

    # Merge and write as a single JSON array file
    merged = build_output_records(sel1) + build_output_records(sel2)
    write_json_array(OUT_PATH, merged)

    # Brief log
    print(f"[OK] File 1 flattened samples: {len(file1_pairs)}, selected: {len(sel1)}")
    print(f"[OK] File 2 flattened samples: {len(file2_pairs)}, selected: {len(sel2)}")
    print(f"[OK] Merged output: {len(merged)} items -> {OUT_PATH}")

if __name__ == "__main__":
    main()