import json
import math
import random
from pathlib import Path
from collections import defaultdict

INPUT_JSONL_PATH = Path("AceReason-1.1-Math.uuid.gt.verify.true.shuf.DeepSeek-R1_Response_Qwen2.5-Math-7B_ppl_list.jsonl")
OUTPUT_JSONL_PATH = Path("AceReason-1.1-Math.uuid.gt.verify.true.shuf.DeepSeek-R1_Response_Qwen2.5-Math-7B_ppl_list.sample_gauss_mean3.0_std0.25_len.5w.1query10response.jsonl")

# Field names
FIELD_QUERY = "query"
FIELD_RESPONSES = "DeepSeek-R1_Response_List"
FIELD_PPL_LIST = "DeepSeek-R1_Response_Qwen2.5-Math-7B_ppl_list"
FIELD_TOKEN_LEN_LIST = "DeepSeek-R1_Response_token_len_list"

# PPL range and bin settings
MIN_PPL = 1.0
MAX_PPL = 5.0
BIN_WIDTH = 0.05

# Target distribution settings (Truncated Normal Distribution)
TARGET_CENTER = 3.0        # Mean (mu)
TARGET_STD = 0.25          # Standard deviation (sigma)
TOTAL_SAMPLES = 50000      # Target total samples
MAX_RESP_PER_QUERY = 10     # Max responses kept per query

# Sampling random seed
RANDOM_SEED = 42

# Strategy toggle
PRIORITIZE_LONG_RESPONSES = True  # True: Prioritize long responses within bin; False: Random within bin

# ====================================================

def get_num_bins():
    """Calculate total number of bins"""
    # Use round to prevent tiny errors caused by floating point division
    return int(round((MAX_PPL - MIN_PPL) / BIN_WIDTH))

def gaussian_pdf(x, mu, sigma):
    """Calculate Gaussian probability density function value"""
    if sigma <= 0:
        return 0.0
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def compute_target_bin_counts(
    total_samples: int,
    target_center: float,
    target_std: float,
    min_ppl: float,
    bin_width: float,
    num_bins: int
):
    """
    Calculate target sample counts for each bin under truncated normal distribution.
    Key point: Normalize after calculating PDF to ensure sum of probabilities is 1.
    """
    print(f"Calculating target distribution (Truncated Normal: mu={target_center}, sigma={target_std}, range=[{min_ppl}, {MAX_PPL}])...")
    
    pdf_values = []
    for i in range(num_bins):
        # Calculate probability density at the center of the bin
        bin_mid = min_ppl + (i + 0.5) * bin_width
        p = gaussian_pdf(bin_mid, target_center, target_std)
        pdf_values.append(p)

    # === Core Logic: Re-normalization ===
    # Since we only take values between [MIN_PPL, MAX_PPL], 
    # we must use the sum of these probability values as the denominator to re-calculate the proportion of each bin.
    total_pdf = sum(pdf_values)
    if total_pdf == 0:
        raise ValueError("Normal distribution calculation results are all 0, please check PPL range or variance settings.")
    
    # Normalized probability list
    probs = [p / total_pdf for p in pdf_values]

    # Calculate theoretical sample count for each bin (float)
    desired = [p * total_samples for p in probs]
    
    # Round down
    bin_counts = [int(math.floor(d)) for d in desired]

    # Handle samples lost after rounding (remainder allocation)
    current_total = sum(bin_counts)
    diff = total_samples - current_total

    if diff > 0:
        # Calculate fractional part
        fractions = [(d - math.floor(d), i) for i, d in enumerate(desired)]
        # Sort by fractional part descending, prioritize adding 1 to bins with larger fractions
        fractions.sort(key=lambda x: x[0], reverse=True)
        
        for i in range(diff):
            idx = fractions[i][1]
            bin_counts[idx] += 1
            
    elif diff < 0:
        # Theoretically floor won't cause diff < 0, but logic kept for robustness
        diff = -diff
        fractions = [(d - math.floor(d), i) for i, d in enumerate(desired)]
        fractions.sort(key=lambda x: x[0]) # Smaller fractions subtracted first
        for i in range(diff):
            idx = fractions[i][1]
            if bin_counts[idx] > 0:
                bin_counts[idx] -= 1

    assert sum(bin_counts) == total_samples, f"Sample allocation calculation error: {sum(bin_counts)} != {total_samples}"
    return bin_counts

def get_bin_index(ppl: float):
    """
    Find corresponding bin index given ppl.
    Added epsilon to prevent floating point boundary errors.
    """
    if ppl < MIN_PPL or ppl > MAX_PPL:
        return None
    
    num_bins = get_num_bins()
    
    # Use epsilon (1e-9) to prevent 1.01 from being calculated as 1.009999 and falling into the previous bin
    idx = int((ppl - MIN_PPL) / BIN_WIDTH + 1e-9)
    
    # Boundary handling: If ppl exactly equals MAX_PPL, assign to the last bin
    if idx >= num_bins:
        idx = num_bins - 1
        
    if idx < 0:
        return None
        
    return idx

def load_data_group_by_bin():
    """Read data and group by bin"""
    bin_to_items = defaultdict(list)
    line_count = 0
    valid_item_count = 0
    num_bins = get_num_bins()

    print(f"Start reading data: {INPUT_JSONL_PATH}")
    with INPUT_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no % 100000 == 0:
                print(f"Processed {line_no} lines...")
                
            line = line.strip()
            if not line: continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            query = obj.get(FIELD_QUERY)
            resp_list = obj.get(FIELD_RESPONSES)
            ppl_list = obj.get(FIELD_PPL_LIST)
            token_len_list = obj.get(FIELD_TOKEN_LEN_LIST)

            if not query or not isinstance(resp_list, list) or not isinstance(ppl_list, list):
                continue

            if not isinstance(token_len_list, list):
                token_len_list = [None] * len(resp_list)

            # Align length
            min_len = min(len(resp_list), len(ppl_list), len(token_len_list))

            for i in range(min_len):
                try:
                    ppl_val = float(ppl_list[i])
                except (TypeError, ValueError):
                    continue
                
                idx = get_bin_index(ppl_val)
                if idx is None:
                    continue # Discard data outside [MIN_PPL, MAX_PPL] range

                # Length handling
                try:
                    t_len = int(token_len_list[i]) if token_len_list[i] is not None else 0
                except:
                    t_len = 0

                bin_to_items[idx].append({
                    "query": query,
                    "response": resp_list[i],
                    "ppl": ppl_val,
                    "len": t_len
                })
                valid_item_count += 1
            
            line_count += 1

    print(f"Reading completed. Lines processed: {line_count}, Valid samples (within PPL range): {valid_item_count}")
    return bin_to_items

def sample_with_query_cap(bin_to_items, target_bin_counts):
    """Execute sampling"""
    num_bins = get_num_bins()
    rnd = random.Random(RANDOM_SEED)

    bin_samples = {}
    per_query_counts = defaultdict(int) # Global query count
    
    total_shortage = 0 # Record number of missing samples

    print("Start sampling by bin...")
    for bin_index in range(num_bins):
        items = bin_to_items.get(bin_index, [])
        target = target_bin_counts[bin_index]

        if target <= 0:
            continue

        # Sorting strategy
        if PRIORITIZE_LONG_RESPONSES:
            # Sort by length descending, random if length is same
            # Trick: key uses tuple (len, random) to shuffle order for same lengths
            items_sorted = sorted(items, key=lambda x: (x["len"], rnd.random()), reverse=True)
        else:
            items_sorted = list(items)
            rnd.shuffle(items_sorted)

        taken_items = []
        local_counts = defaultdict(int) # Query count within current bin

        # Multi-round filtering: First try taking 1 per query, if not enough take 2nd...
        # Until target is reached or MAX_RESP_PER_QUERY is reached
        for local_cap in range(1, MAX_RESP_PER_QUERY + 1):
            if len(taken_items) >= target:
                break
            
            for it in items_sorted:
                if len(taken_items) >= target:
                    break
                
                q = it["query"]
                
                # Check global limit
                if per_query_counts[q] >= MAX_RESP_PER_QUERY:
                    continue
                
                # Check round limit within current bin (to ensure samples come from different queries as much as possible)
                if local_counts[q] >= local_cap:
                    continue

                # Selected
                per_query_counts[q] += 1
                local_counts[q] += 1
                taken_items.append(it)

        bin_samples[bin_index] = taken_items
        
        # Check if target is met
        if len(taken_items) < target:
            shortage = target - len(taken_items)
            total_shortage += shortage
            # Only print when shortage is large to avoid spamming
            if shortage > 0: 
                # Calculate PPL range corresponding to bin for prompt
                b_start = MIN_PPL + bin_index * BIN_WIDTH
                b_end = b_start + BIN_WIDTH
                print(f"[Warning] Bin {bin_index} ({b_start:.2f}-{b_end:.2f}) insufficient data! Target: {target}, Actual: {len(taken_items)}, Shortage: {shortage}")

    # Aggregate
    all_samples = []
    for b_idx, items in bin_samples.items():
        for it in items:
            all_samples.append(it)

    rnd.shuffle(all_samples)
    
    print("-" * 30)
    print(f"Sampling finished.")
    print(f"Target total: {TOTAL_SAMPLES}")
    print(f"Actual total: {len(all_samples)}")
    if total_shortage > 0:
        print(f"[Severe Warning] Total missing {total_shortage} data. This means valid data in source within specific PPL range is less than theoretical demand of normal distribution.")
    else:
        print("Successfully met all sampling requirements.")
    print("-" * 30)
    
    return all_samples

def save_samples_to_jsonl(samples):
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to file: {OUTPUT_JSONL_PATH}")
    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as f:
        for item in samples:
            out_obj = {
                "query": item["query"],
                "response": item["response"],
                "ppl": item["ppl"],
                "len": item["len"],
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    print("Writing completed.")

def main():
    num_bins = get_num_bins()
    print(f"Config info: PPL range [{MIN_PPL}, {MAX_PPL}], Bin width {BIN_WIDTH}, Total bins {num_bins}")
    
    # 1. Calculate target distribution (normalization logic included)
    target_bin_counts = compute_target_bin_counts(
        total_samples=TOTAL_SAMPLES,
        target_center=TARGET_CENTER,
        target_std=TARGET_STD,
        min_ppl=MIN_PPL,
        bin_width=BIN_WIDTH,
        num_bins=num_bins
    )
    
    # 2. Read data
    bin_to_items = load_data_group_by_bin()
    
    # 3. Sampling
    final_samples = sample_with_query_cap(bin_to_items, target_bin_counts)
    
    # 4. Save
    save_samples_to_jsonl(final_samples)

if __name__ == "__main__":
    main()