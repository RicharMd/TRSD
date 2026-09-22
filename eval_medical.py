import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_VERIFIER_PATH = "model/medical_o1_verifier_3B"
DEFAULT_EVAL_DATA = "data/medical_data/eval_data"

VERIFIER_TEMPLATE = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False" . {}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on the medical verifiable holdout set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained policy model")
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=DEFAULT_EVAL_DATA,
        help="On-disk medical eval dataset (prompt + answer)",
    )
    parser.add_argument(
        "--verifier_path",
        type=str,
        default=DEFAULT_VERIFIER_PATH,
        help="Local Huatuo medical_o1_verifier_3B outcome RM",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Generation cap for the policy model")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 for greedy decoding")
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to model_path")
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory fraction for the policy model",
    )
    parser.add_argument(
        "--verifier_batch_size",
        type=int,
        default=16,
        help="Batch size for the 3B verifier forward pass",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only run vLLM generation and save responses; skip verifier scoring",
    )
    return parser.parse_args()


def load_policy_model(model_path: str, gpu_memory_utilization: float):
    print(f"Loading policy model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=4096,
        trust_remote_code=True,
    )
    return llm, tokenizer


def load_eval_data(path: str) -> Dataset:
    print(f"Loading medical eval dataset from {path}")
    return Dataset.load_from_disk(path)


def generate_responses(llm, tokenizer, prompts, max_new_tokens: int, temperature: float):
    formatted_prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for prompt in prompts
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def extract_answer_text(response: str) -> str:
    """Prefer <answer>...</answer>; otherwise use the full response for the verifier."""
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].split("</answer>")[0]
        stripped = answer.strip()
        if stripped:
            return stripped
    return response.strip()


def load_verifier(verifier_path: str):
    print(f"Loading verifier from {verifier_path}")
    tokenizer = AutoTokenizer.from_pretrained(verifier_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        verifier_path,
        torch_dtype="auto",
        device_map="auto",
        num_labels=2,
    )
    model.eval()
    return model, tokenizer


def build_verifier_inputs(model_response: str, reference_answer: str, eos_token: str) -> str:
    return VERIFIER_TEMPLATE.format(model_response, reference_answer, eos_token)


@torch.inference_mode()
def score_with_verifier(model, tokenizer, model_responses, reference_answers, batch_size: int):
    scores = []
    verdicts = []
    eos_token = tokenizer.eos_token or ""

    for start in range(0, len(model_responses), batch_size):
        batch_responses = model_responses[start : start + batch_size]
        batch_refs = reference_answers[start : start + batch_size]
        texts = [
            build_verifier_inputs(response, reference, eos_token)
            for response, reference in zip(batch_responses, batch_refs)
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        logits = model(**encoded, return_dict=True).logits
        probabilities = F.softmax(logits, dim=-1)
        batch_correct = (probabilities[:, 1] > 0.5).tolist()
        scores.extend(int(correct) for correct in batch_correct)
        verdicts.extend("True" if correct else "False" for correct in batch_correct)

    return scores, verdicts


def main():
    args = parse_args()
    output_dir = args.output_dir or args.model_path
    os.makedirs(output_dir, exist_ok=True)

    llm, policy_tokenizer = load_policy_model(args.model_path, args.vllm_gpu_memory_utilization)
    eval_data = load_eval_data(args.eval_data_path)

    prompts = [example["prompt"] for example in eval_data]
    references = [example["answer"] for example in eval_data]
    source_indices = [example.get("source_index", idx) for idx, example in enumerate(eval_data)]

    responses = generate_responses(
        llm,
        policy_tokenizer,
        prompts,
        args.max_new_tokens,
        args.temperature,
    )
    extracted_answers = [extract_answer_text(response) for response in responses]

    scores = None
    verdicts = None
    accuracy = None
    if not args.generate_only:
        verifier_model, verifier_tokenizer = load_verifier(args.verifier_path)
        scores, verdicts = score_with_verifier(
            verifier_model,
            verifier_tokenizer,
            extracted_answers,
            references,
            args.verifier_batch_size,
        )
        accuracy = float(np.mean(scores))

        print("\n" + "=" * 60)
        print("Medical Eval (verifier_3B outcome RM)")
        print(f"  Total samples: {len(scores)}")
        print(f"  Correct: {sum(scores)}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("=" * 60)

    results_to_save = {
        "accuracy": accuracy,
        "num_correct": int(sum(scores)) if scores is not None else None,
        "num_total": len(responses),
        "metric": "medical_o1_verifier_3B_correct",
        "per_sample_scores": scores,
        "config": {
            "model_path": args.model_path,
            "eval_data_path": args.eval_data_path,
            "verifier_path": args.verifier_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "generate_only": args.generate_only,
        },
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results_to_save, handle, indent=2)
    print(f"\nSaved results to {results_path}")

    responses_path = os.path.join(output_dir, "eval_responses.json")
    with open(responses_path, "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "index": i,
                    "source_index": source_indices[i],
                    "prompt": prompts[i],
                    "response": responses[i],
                    "extracted_answer": extracted_answers[i],
                    "reference_answer": references[i],
                    "verifier_verdict": verdicts[i] if verdicts is not None else None,
                    "correct": bool(scores[i]) if scores is not None else None,
                }
                for i in range(len(responses))
            ],
            handle,
            indent=2,
        )
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()
