import sys
import json
import argparse
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from deepscaler.rewards.math_reward import deepscaler_reward_fn
import traceback
import os

suffix_1 = (
    " Let's think step by step and output the final answer within \\boxed{}."  # original suffix
)
suffix_2 = (
    "\nLet's reason step by step. Enclose the reasoning process within <think>...</think>, then summarize it and present the final answer within \\boxed{} — for example: <think>reasoning process here</think> \\boxed{answer here}."
)


def request_model(client, prompt, seed, model):
    for _ in range(10):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"add_generation_prompt": True, "seed": seed},
                temperature=0.6,
                top_p=0.95,
                timeout=100000000,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            traceback.print_exc()
            
    print("Fatal Error: Request failed 10 times. Exiting program immediately.")
    os._exit(1)


def main(seed):
    parser = argparse.ArgumentParser(description="vLLM inference and statistics for pass@1 and pass@K")
    parser.add_argument('--model', type=str, required=True, help="vLLM model name")
    parser.add_argument('--file', type=str, required=True, help="Path to the test set JSON file")
    parser.add_argument('--ports', type=str, required=True, help="List of vLLM server port numbers, separated by commas")
    parser.add_argument('--repeat', type=int, required=True, help="Number of repetitions per question (e.g., 64)")
    parser.add_argument('--concurrency', type=int, required=True, help="Maximum concurrent requests per port X")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    args = parser.parse_args()
    suffix = suffix_2

    ports = args.ports.split(',')
    client_list = []
    for port in ports:
        openai_api_key = "sk-xxx"
        openai_api_base = f"http://localhost:{port}/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        sem = threading.Semaphore(args.concurrency)
        client_list.append((client, sem))
    total_clients = len(client_list)
    total_concurrency = total_clients * args.concurrency
    print(f"A total of {total_clients} clients are constructed, and the total concurrency is {total_concurrency}.")

    with open(args.file, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    if not original_data:
        print("Test set is empty.")
        return

    tasks = []
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        if q_idx == 0:
            print(f"Question sample:{prompt}, Answer sample:{ground_truth}")
        for rep in range(args.repeat):
            tasks.append((q_idx, prompt, ground_truth, rep))

    detailed_results = {}
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        detailed_results[q_idx] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "responses": [None] * args.repeat,
            "correct_flags": [False] * args.repeat
        }

    total_responses = len(tasks)
    correct_count = 0

    executor = ThreadPoolExecutor(max_workers=total_concurrency)
    future_list = []
    global_index = 0

    def request_task(client, semaphore, prompt, seed, model):
        with semaphore:
            return request_model(client, prompt, seed, model)

    for task in tasks:
        q_idx, prompt, ground_truth, rep = task
        client, sem = client_list[global_index % total_clients]
        global_index += 1
        fut = executor.submit(request_task, client, sem, prompt, seed, args.model)
        future_list.append((q_idx, rep, prompt, ground_truth, fut))

    for q_idx, rep, prompt, ground_truth, fut in future_list:
        model_response = fut.result() 

        processed_response = model_response.replace("\n", "")
        try:
            is_correct = deepscaler_reward_fn(model_response, ground_truth)
        except Exception as e:
            is_correct = False

        detailed_results[q_idx]["responses"][rep] = processed_response
        detailed_results[q_idx]["correct_flags"][rep] = is_correct

        if is_correct:
            correct_count += 1

    pass_at_1 = correct_count / total_responses

    pass16_correct = 0
    for q_idx, result in detailed_results.items():
        question_correct = any(result["correct_flags"])
        result["is_correct"] = question_correct
        result["question_accuracy"] = sum(result["correct_flags"]) / args.repeat
        if question_correct:
            pass16_correct += 1
    total_questions = len(original_data)
    pass_at_16 = pass16_correct / total_questions

    summary_results = {
        "file": args.file,
        "model": args.model,
        "pass@1": pass_at_1,
        f"pass@{args.repeat}": pass_at_16,
        "total_responses": total_responses,
        "total_questions": total_questions
    }
    with open(args.output_dir + "/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    detailed_results_list = []
    for q_idx in sorted(detailed_results.keys()):
        detailed_results_list.append(detailed_results[q_idx])
    with open(args.output_dir + "/results_details.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results_list, f, ensure_ascii=False, indent=2)

    print("######### Deepscaler Statistics Results:")
    print(f"pass@1 = {pass_at_1:.4f}, pass@{args.repeat} = {pass_at_16:.4f}")
    print(f"Statistics have been saved to {args.output_dir}/results_summary.json and {args.output_dir}/results_details.json")

if __name__ == "__main__":
    main(None)
