#!/usr/bin/env python
# coding: utf-8
"""
Compute PPL for every response string in DeepSeek-R1_Response_List (jsonl).

用法
python ppl_single_gpu.py <part_idx> <input_dir> <output_dir> <model_path> <max_seq_len> <gpu_id> <batch_size>

• part_idx   : 数据分片编号
• input_dir  : 输入目录
• output_dir : 结果写入目录
• model_path : HF / 4bit / GPTQ 路径均可
• max_seq_len: tokenizer 截断长度
• gpu_id     : 指定在哪张 GPU 上推断
• batch_size : 尝试的初始 batch size（遇到 OOM 会自动回退）

说明
- 为避免 OOM，本脚本包含三重兜底：
  1) 发生 OOM 后进行彻底的显存回收（gc + empty_cache + ipc_collect）；
  2) 对当前批进行“二分回退”的动态缩小 batch；
  3) 若单条仍 OOM，则改为“序列分块”前向（优先使用 past_key_values），在 L 维切片累计 NLL。
- 可通过 step_tokens 调整序列分块的块大小（默认 128）。越小越省显存但更慢。
"""

# ---------- 建议的 allocator 配置（尽量放在导入 torch 之前） ----------
import os
# os.environ.setdefault(
#     "PYTORCH_CUDA_ALLOC_CONF",
#     "garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
# )

import sys, json, math, gc
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True   # TF32 = True → matmul 更快

# -------------------- 工具函数 --------------------
def hard_cuda_cleanup():
    """更彻底的清理：Python GC + 释放缓存 + 回收 IPC 残留。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def count_tokens(texts: List[str], tokenizer) -> List[int]:
    enc = tokenizer(texts, return_tensors="pt",
                    padding=True, truncation=False)
    return enc.attention_mask.sum(dim=1).tolist()

@torch.inference_mode()
def ppl_forward(texts: List[str],
                tokenizer,
                model,
                max_seq_len: int,
                dtype) -> Tuple[List[float], List[int]]:
    """
    标准一次性前向：会产生 (B, L, V) 的 logits。速度最快，但峰值显存最大。
    """
    token_lens = count_tokens(texts, tokenizer)
    enc = tokenizer(texts, return_tensors="pt",
                    padding=True, truncation=True,
                    max_length=max_seq_len).to(model.device)

    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(**enc).logits          # (B, L, V)

    # shift
    logits = logits[:, :-1].contiguous()
    labels = enc["input_ids"][:, 1:].contiguous()
    mask   = enc["attention_mask"][:, 1:].bool()

    loss_tok = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction='none'
    ).view_as(labels)

    ppl_list = []
    for i in range(loss_tok.size(0)):
        denom = mask[i].sum()
        if denom == 0:
            ppl_list.append(float("nan"))
        else:
            nll = (loss_tok[i] * mask[i]).sum() / denom
            ppl_list.append(math.exp(nll.item()))

    # 释放
    del logits, loss_tok, enc, labels, mask
    hard_cuda_cleanup()
    return ppl_list, token_lens

@torch.inference_mode()
def ppl_forward_chunked_single(text: str,
                               tokenizer,
                               model,
                               max_seq_len: int,
                               dtype,
                               step_tokens: int = 128,
                               ctx_window: int = None) -> Tuple[List[float], List[int]]:
    """
    对单条样本按序列维分块，优先使用 past_key_values。
    - step_tokens: 每次前向送入多少 token，减小峰值 (chunk_len * V)。
    - ctx_window:  可选，有限上下文窗口大小。若指定，将仅保留最近 ctx_window 个 token 作为条件，
                  可进一步降低 KV 占用；但得到的是“有限上下文”近似 PPL。

    返回: ([ppl], [token_len])
    """
    print("WARNING Proc {PART_IDX} called ppl_forward_chunked_single.")
    enc = tokenizer([text], return_tensors="pt",
                    padding=False, truncation=True, max_length=max_seq_len).to(model.device)
    input_ids = enc["input_ids"]  # (1, L)
    attn_mask = enc["attention_mask"]  # (1, L)
    L = input_ids.size(1)

    if L <= 1:
        # 文本太短，ppl 定义不稳定，直接返回 nan
        del enc
        return [float("nan")], [L]

    total_nll = 0.0
    total_cnt = 0

    # 检测是否支持 cache
    use_cache = True
    past = None

    # 预热一个 token，建立 KV
    try:
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids[:, :1], attention_mask=attn_mask[:, :1], use_cache=True)
        past = getattr(out, "past_key_values", None)
        if past is None:
            use_cache = False
        del out
    except Exception:
        use_cache = False
        past = None
        hard_cuda_cleanup()

    pos = 1
    while pos < L:
        end = min(L, pos + step_tokens)

        if ctx_window is not None:
            # 有限窗口：保留最近 ctx_window 个 token 作为上下文
            # 对于使用 cache 的情况，可以周期性重建 cache 以限制 KV 体积
            window_start = max(0, end - ctx_window)
        else:
            window_start = 0

        if use_cache:
            # 使用 cache：只有第一次送 1 token 建立 cache，后续送增量块
            # 但若设置 ctx_window，达到窗口边界时重建 cache
            if ctx_window is not None and window_start > 0 and window_start >= pos:
                # 重建：用窗口的起点重新热身
                past = None
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model(input_ids[:, window_start:window_start+1],
                                attention_mask=attn_mask[:, window_start:window_start+1],
                                use_cache=True)
                past = out.past_key_values
                del out
                hard_cuda_cleanup()

                # 从 window_start+1 开始逐块推进到 end
                rebuild_pos = window_start + 1
                while rebuild_pos < end:
                    rebuild_end = min(end, rebuild_pos + step_tokens)
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        out = model(input_ids[:, rebuild_pos:rebuild_end],
                                    attention_mask=attn_mask[:, window_start:rebuild_end],
                                    past_key_values=past, use_cache=True)
                    past = out.past_key_values

                    logits = out.logits  # (1, chunk_len, V)
                    labels = input_ids[:, rebuild_pos:rebuild_end]

                    # 对齐：logits 预测当前位置 token，与 labels 对齐即可
                    logprobs = torch.log_softmax(logits, dim=-1)
                    nll = F.nll_loss(
                        logprobs.view(-1, logprobs.size(-1)),
                        labels.view(-1),
                        reduction='sum'
                    )
                    cnt = (rebuild_end - rebuild_pos)
                    total_nll += nll.item()
                    total_cnt += cnt

                    del logits, labels, logprobs, nll, out
                    hard_cuda_cleanup()
                    rebuild_pos = rebuild_end

                pos = end
                continue

            # 正常增量推进
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(input_ids[:, pos:end],
                            attention_mask=attn_mask[:, :end],
                            past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits
            labels = input_ids[:, pos:end]
        else:
            # 不使用 cache：需要提供完整上下文（或有限窗口）
            ctx_start = window_start
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(input_ids[:, ctx_start:end],
                            attention_mask=attn_mask[:, ctx_start:end],
                            use_cache=False)
            logits = out.logits  # (1, end-ctx_start, V)
            # 需要取与 positions pos..end-1 对应的 logits 切片
            offset = pos - ctx_start
            logits = logits[:, offset:, :]           # (1, chunk_len, V)
            labels = input_ids[:, pos:end]

        logprobs = torch.log_softmax(logits, dim=-1)
        nll = F.nll_loss(
            logprobs.view(-1, logprobs.size(-1)),
            labels.view(-1),
            reduction='sum'
        )
        cnt = (end - pos)
        total_nll += nll.item()
        total_cnt += cnt

        del logits, labels, logprobs, nll, out
        hard_cuda_cleanup()
        pos = end

    ppl = math.exp(total_nll / max(total_cnt, 1))
    token_len = L

    del input_ids, attn_mask, enc, past
    hard_cuda_cleanup()
    return [ppl], [token_len]

def safe_ppl_with_backoff(texts: List[str],
                          tokenizer,
                          model,
                          max_seq_len: int,
                          dtype,
                          step_tokens: int = 128,
                          min_bs: int = 1) -> Tuple[List[float], List[int]]:
    """
    先尝试整批 ppl_forward；OOM 则二分回退 batch；
    如果缩到单条仍 OOM，则改用序列分块 ppl_forward_chunked_single。
    """
    n = len(texts)
    ppl_acc: List[float] = []
    len_acc: List[int] = []

    i = 0
    while i < n:
        # 动态确定本轮 batch 大小（尽量大，失败就对半）
        bs_try = n - i
        while bs_try >= min_bs:
            try:
                p, l = ppl_forward(texts[i:i+bs_try], tokenizer, model, max_seq_len, dtype)
                ppl_acc.extend(p); len_acc.extend(l)
                i += bs_try
                break
            except torch.cuda.OutOfMemoryError:
                hard_cuda_cleanup()
                # 二分回退
                if bs_try == 1:
                    # 单条仍失败 → 改用序列分块
                    try:
                        p, l = ppl_forward_chunked_single(
                            texts[i], tokenizer, model, max_seq_len, dtype,
                            step_tokens=step_tokens, ctx_window=None
                        )
                        ppl_acc.extend(p); len_acc.extend(l)
                        i += 1
                        break
                    except torch.cuda.OutOfMemoryError:
                        # 仍 OOM，进一步缩小 step_tokens 或启用有限窗口
                        hard_cuda_cleanup()
                        p, l = ppl_forward_chunked_single(
                            texts[i], tokenizer, model, max_seq_len, dtype,
                            step_tokens=max(32, step_tokens // 2),
                            ctx_window=None  # 根据显存再调小
                        )
                        ppl_acc.extend(p); len_acc.extend(l)
                        i += 1
                        break
                else:
                    bs_try = max(min_bs, bs_try // 2)
    return ppl_acc, len_acc

# --------------------------- 主流程 ----------------------------- #
if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python ppl_single_gpu.py <part_idx> <input_dir> <output_dir> "
              "<model_path> <max_seq_len> <gpu_id> <batch_size>")
        sys.exit(1)

    PART_IDX       = int(sys.argv[1])
    INPUT_DIR      = sys.argv[2]
    OUTPUT_DIR     = sys.argv[3]
    MODEL_PATH     = sys.argv[4]
    MAX_SEQ_LEN    = int(sys.argv[5])
    GPU_ID         = int(sys.argv[6])
    BATCH_SIZE     = int(sys.argv[7])  # 初始尝试的 batch size

    ROW_BATCH      = 1                  # 一次拿多少行 jsonl 一起算
    DTYPE          = torch.bfloat16     # bf16 更稳
    STEP_TOKENS    = 8192               # 序列分块大小（可按显存调节）

    infile  = os.path.join(INPUT_DIR, f"part_{PART_IDX:04d}.jsonl")
    outfile = os.path.join(OUTPUT_DIR, f"P{PART_IDX}.jsonl")
    if not os.path.exists(infile):
        print(f"[Proc {PART_IDX}] File not found: {infile}")
        sys.exit(0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读数据
    data = [json.loads(l) for l in open(infile, encoding="utf-8")]

    # 断点续跑
    completed = 0
    if os.path.exists(outfile):
        completed = sum(1 for _ in open(outfile, encoding="utf-8"))
    print(f"[Proc {PART_IDX}] Completed lines: {completed}/{len(data)}")

    # GPU & 模型
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f"cuda:{GPU_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True,
                                              padding_side='right')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (AutoModelForCausalLM
             .from_pretrained(MODEL_PATH,
                              torch_dtype=DTYPE,
                              device_map={"": device.index},
                              low_cpu_mem_usage=True)
             .eval())

    hard_cuda_cleanup()

    # -------------------- 主循环（按 ROW_BATCH 做大 batch） --------------------
    with open(outfile, "a", encoding="utf-8") as fout:
        total = len(data)
        for start_row in tqdm(range(completed, total, ROW_BATCH),
                              desc=f"Proc {PART_IDX}"):
            rows = data[start_row: start_row + ROW_BATCH]

            # 1) 把 ROW_BATCH 行的 resp_list 合并
            global_resps = []
            map_idx = []  # 记录各 row 里 resp 的数量，用于回拆
            for r in rows:
                lst = r.get("DeepSeek-R1_Response_List", [])
                map_idx.append(len(lst))
                if lst:
                    global_resps.extend(lst)

            # 2) 统一推断（内置动态回退 + 序列分块兜底）
            ppl_all, len_all = [], []
            # 分块送入，避免一次性 texts 太大
            for s in range(0, len(global_resps), BATCH_SIZE):
                sub = global_resps[s:s + BATCH_SIZE]
                p_sub, l_sub = safe_ppl_with_backoff(
                    sub, tokenizer, model, MAX_SEQ_LEN, DTYPE,
                    step_tokens=STEP_TOKENS, min_bs=1
                )
                ppl_all.extend(p_sub)
                len_all.extend(l_sub)

            # 3) 把结果按 map_idx 回拆到各行
            cursor = 0
            for row, n_resp in zip(rows, map_idx):
                if n_resp == 0:
                    row["r1_response_qwen3_1.7b_ppl_list"] = []
                    row["r1_response_qwen3_1.7b_token_len_list"] = []
                else:
                    row["r1_response_qwen3_1.7b_ppl_list"] = ppl_all[cursor: cursor + n_resp]
                    row["r1_response_qwen3_1.7b_token_len_list"] = len_all[cursor: cursor + n_resp]
                cursor += n_resp
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"[Proc {PART_IDX}] All done! Output → {outfile}")
