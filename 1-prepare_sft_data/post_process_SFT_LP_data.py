import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

INPUT_PATH  = Path("AceReason-1.1-Math.uuid.gt.verify.true.shuf.r1_response_Qwen2.5-Math-1.5B.minlen10000_lowppl_50000.jsonl")   # Original JSONL file
OUTPUT_PATH = Path("AceReason-1.1-Math.uuid.gt.verify.true.shuf.r1_response_Qwen2.5-Math-1.5B.minlen10000_lowppl_50000.alpaca.json")   # Target JSON file
SUFFIX = "\nLet's reason step by step. Enclose the reasoning process within <think>...</think>, then summarize it and present the final answer within \\boxed{} — for example: <think>reasoning process here</think> \\boxed{answer here}."


def transform_line(line: str) -> list[dict]:
    """
    Split a line of original JSON into multiple target JSON items.
    The return value is a list, because one line may correspond to multiple responses.
    """
    line = line.strip()
    if not line:
        return []

    data = json.loads(line)
    query     = data.get("query", "")
    responses = data.get("DeepSeek-R1_Response_List", []) or []
    query = query + SUFFIX

    return [
        {
            "instruction": query,
            "input": "",
            "output": resp,
            "history": [],
            "loss_type": 0,
        }
        for resp in responses
    ]


def main() -> None:
    # Read all lines
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Multi-process conversion
    with Pool(processes=cpu_count()) as pool:
        all_chunks = pool.map(transform_line, lines)

    # Flatten into a list
    flattened: list[dict] = [item for chunk in all_chunks for item in chunk]

    # Write out JSON array
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=2)

    print(f"✅ Conversion completed, generated {len(flattened)} items -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()