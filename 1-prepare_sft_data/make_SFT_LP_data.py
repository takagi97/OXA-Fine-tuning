import json
import heapq
import os
from collections import defaultdict

# === Configuration Area ===
INPUT_PATH = "AceReason-1.1-Math.uuid.gt.verify.true.shuf.r1_response_qwen3_1.7b_ppl_list.jsonl"
LOWEST_OUT_PATH = "AceReason-1.1-Math.uuid.gt.verify.true.shuf.r1_response_qwen3_1.7b_ppl_list.lowest_50000.1query1response.jsonl"

GLOBAL_K = 50000  # Globally select the lowest 50000 Responses by PPL
MAX_RESPONSES_PER_QUERY = 1
PPL_KEY = "r1_response_qwen3_1.7b_ppl_list"
RESPONSE_KEY = "DeepSeek-R1_Response_List" 
# =================

def main():
    os.makedirs(os.path.dirname(LOWEST_OUT_PATH), exist_ok=True)

    # Max heap (used to keep the smallest K PPLs)
    # Heap stores: (-ppl, counter, base_record_str, response_text, ppl_val)
    lowest_heap = [] 
    
    counter = 0
    total_lines = 0
    total_responses_scanned = 0
    skipped = 0

    print(f"[INFO] Reading file: {INPUT_PATH}")
    print(f"[CONFIG] Global target quantity: {GLOBAL_K}")
    print(f"[CONFIG] Max responses per query limit: {MAX_RESPONSES_PER_QUERY}")

    with open(INPUT_PATH, "r", encoding="utf-8") as fin:
        for line in fin:
            total_lines += 1
            line = line.strip()
            if not line: continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            
            ppl_list = record.get(PPL_KEY)
            response_list = record.get(RESPONSE_KEY)

            # Data validation
            if not isinstance(ppl_list, list) or not isinstance(response_list, list):
                if total_lines == 1:
                    print(f"[WARN] List fields not found in line 1, please check RESPONSE_KEY: {RESPONSE_KEY}")
                skipped += 1
                continue
            
            if len(ppl_list) != len(response_list):
                skipped += 1
                continue

            # 1. Prepare base info (remove two large lists, prepare for aggregation)
            base_info = record.copy()
            del base_info[PPL_KEY]
            del base_info[RESPONSE_KEY]
            base_json_str = json.dumps(base_info, ensure_ascii=False)

            # 2. [Local Processing] Pair Response and PPL within the line
            local_pairs = []
            for i in range(len(ppl_list)):
                try:
                    p = float(ppl_list[i])
                    r = response_list[i]
                    local_pairs.append((p, r))
                except (ValueError, TypeError):
                    continue
            
            total_responses_scanned += len(local_pairs)

            if not local_pairs:
                continue

            # 3. [Local Filtering] Sort internally within this line first, take only the best N
            # The logic here is: if this line has 10 responses, we only allow the best N to participate in global competition
            # This physically ensures that in the final result, this line has at most N items
            local_pairs.sort(key=lambda x: x[0]) # Sort by PPL from low to high
            best_candidates = local_pairs[:MAX_RESPONSES_PER_QUERY]

            # 4. [Global Competition] Put selected candidates into global heap
            for p_val, r_text in best_candidates:
                counter += 1
                heap_item = (-p_val, counter, base_json_str, r_text, p_val)

                if len(lowest_heap) < GLOBAL_K:
                    heapq.heappush(lowest_heap, heap_item)
                else:
                    # If current PPL is smaller (better) than the worst (largest) PPL in the heap
                    # lowest_heap[0][0] stores -MaxPPL
                    if -p_val > lowest_heap[0][0]:
                        heapq.heapreplace(lowest_heap, heap_item)

            if total_lines % 50000 == 0:
                print(f" ...Scanned {total_lines} lines, total scanned Responses {total_responses_scanned}...")

    print(f"[INFO] Scanning finished. Total queries: {total_lines}")
    print(f"[INFO] Reorganizing data (Group by query)...")

    # === Reorganization Phase ===
    # Data structure: { base_json_str: { "resps": [], "ppls": [] } }
    grouped_data = defaultdict(lambda: {"resps": [], "ppls": []})

    # Sort heap data first to ensure ppl in the same list are ordered (low to high) when outputting
    sorted_items = sorted(lowest_heap, key=lambda x: -x[0]) 

    for _, _, base_str, resp, ppl_val in sorted_items:
        grouped_data[base_str]["resps"].append(resp)
        grouped_data[base_str]["ppls"].append(ppl_val)

    # === Writing Phase ===
    print(f"[INFO] Writing file: {LOWEST_OUT_PATH}")
    write_count = 0
    
    with open(LOWEST_OUT_PATH, "w", encoding="utf-8") as fout:
        for base_str, lists in grouped_data.items():
            # Check again (theoretically guaranteed by logic above, but as a double safety)
            if len(lists["resps"]) > MAX_RESPONSES_PER_QUERY:
                # Truncate (usually won't happen unless heap logic is wrong, here as a safety net)
                lists["resps"] = lists["resps"][:MAX_RESPONSES_PER_QUERY]
                lists["ppls"] = lists["ppls"][:MAX_RESPONSES_PER_QUERY]
            
            out_record = json.loads(base_str)
            out_record[RESPONSE_KEY] = lists["resps"]
            out_record[PPL_KEY] = lists["ppls"]
            
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            write_count += 1

    print(f"[INFO] Writing completed.")
    print(f"   - Final included Query count: {write_count}")
    print(f"   - Final included Response total: {len(sorted_items)}")
    
if __name__ == "__main__":
    main()