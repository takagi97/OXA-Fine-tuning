import json

# ======= Configuration Area =======
INPUT_JSONL_PATH = "AceReason-1.1-Math.uuid.gt.verify.true.shuf.DeepSeek-R1_Response_Qwen2.5-Math-7B_ppl_list.sample_gauss_mean3.0_std0.25_len.25w.1query10response.jsonl"   # Path to jsonl file (Figure 2 format)
OUTPUT_JSON_PATH = "AceReason-1.1-Math.uuid.gt.verify.true.shuf.DeepSeek-R1_Response_Qwen2.5-Math-7B_ppl_list.sample_gauss_mean3.0_std0.25_len.25w.1query10response.alpaca.json"   # Path to json file (Figure 1 format)

FIXED_INPUT = ""        # "input" field fixed as empty string
FIXED_HISTORY = []      # "history" field fixed as empty list
FIXED_LOSS_TYPE = 0     # "loss_type" field fixed as 0
SUFFIX = "\nLet's reason step by step. Enclose the reasoning process within <think>...</think>, then summarize it and present the final answer within \\boxed{} — for example: <think>reasoning process here</think> \\boxed{answer here}."
# ========================================


def convert_jsonl_to_json():
    converted = []

    # Read jsonl, each line is an independent json
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Assume original field names are "query" and "response"
            instruction = record.get("query", "")
            output = record.get("response", "")

            new_item = {
                "instruction": instruction + SUFFIX,
                "input": FIXED_INPUT,
                "output": output,
                "history": FIXED_HISTORY,
                "loss_type": FIXED_LOSS_TYPE,
            }

            converted.append(new_item)

    # Write as a large json array
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    convert_jsonl_to_json()
    print(f"Conversion completed, result saved to {OUTPUT_JSON_PATH}")