#!/usr/bin/env python3
"""
ablation_context_truncation.py — Context Truncation Ablation for OLMo-3 Degeneration

Tests whether OLMo-3 degeneration is caused by context saturation by limiting
how much conversation history each agent sees.

Conditions:
  - full: complete conversation history (baseline, should degenerate)
  - last_3: only last 3 rounds of history
  - last_1: only the most recent round
  - summary: full history replaced by a 2-sentence summary each round

If context truncation eliminates degeneration, the mechanism is context saturation.
If degeneration persists even with last_1, the issue is something else
(e.g., model-internal state, repetition tendency independent of context length).

Usage:
    python ablation_context_truncation.py \
        --model allenai/OLMo-3-7B-Instruct \
        --output_dir ./ablation_context_truncation \
        --reps 5

Author: Herald (Paper A — context truncation ablation for §4.1 investigation)
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TASKS = {
    "ethical_dilemma": (
        "An AI system can predict criminal recidivism with 85% accuracy. "
        "Should it be used in sentencing decisions? Argue your position."
    ),
    "startup_ideation": (
        "Propose a viable startup that addresses problems in online education. "
        "Describe the core product, target market, and business model."
    ),
    "creative_problem": (
        "A city has 1 million people but can only build 3 new buildings. "
        "What should they build to maximize quality of life? Justify your choices."
    ),
}

SYSTEM_PROMPT = (
    "You are a helpful assistant participating in a brainstorming discussion. "
    "Think step by step, then give your argument."
)

CONTEXT_CONDITIONS = ["full", "last_3", "last_1"]


def load_model(model_name: str):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


def truncate_history(history: list[dict], condition: str) -> list[dict]:
    """Truncate conversation history based on condition."""
    if condition == "full":
        return history
    elif condition == "last_3":
        # Keep only the last 3 rounds (6 turns: 3 from each agent)
        return history[-6:]
    elif condition == "last_1":
        # Keep only the last round (2 turns)
        return history[-2:]
    else:
        raise ValueError(f"Unknown condition: {condition}")


def build_prompt(system_prompt: str, task_prompt: str, history: list[dict],
                 agent_id: str) -> str:
    """Build a chat-format prompt from history."""
    parts = [f"System: {system_prompt}\n\nTopic: {task_prompt}\n"]
    for turn in history:
        speaker = turn["agent_id"]
        parts.append(f"\n{speaker}: {turn['text']}")
    parts.append(f"\n{agent_id}:")
    return "".join(parts)


def generate(model, tokenizer, prompt: str, temperature: float = 0.7,
             max_new_tokens: int = 512) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_k=50,
        )
    elapsed = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, elapsed


def detect_degeneration(text: str, threshold: float = 0.30) -> tuple[bool, float]:
    """Check if top single-word frequency exceeds threshold."""
    words = text.lower().split()
    if not words:
        return True, 1.0
    counts = Counter(words)
    top_freq = counts.most_common(1)[0][1] / len(words)
    return top_freq > threshold, top_freq


def run_ablation(model, tokenizer, task_name: str, task_prompt: str,
                 condition: str, n_rounds: int = 10) -> dict:
    """Run a single debate session with given context truncation condition."""
    history = []
    rounds_data = []

    for r in range(n_rounds):
        round_turns = []
        for agent_id in ["agent_0", "agent_1"]:
            visible_history = truncate_history(history, condition)
            prompt = build_prompt(SYSTEM_PROMPT, task_prompt, visible_history, agent_id)

            text, gen_time = generate(model, tokenizer, prompt)
            is_degen, top_freq = detect_degeneration(text)

            turn = {
                "agent_id": agent_id,
                "round": r,
                "text": text,
                "gen_time_s": round(gen_time, 2),
                "top_word_freq": round(top_freq, 4),
                "is_degenerate": is_degen,
                "prompt_tokens": len(prompt.split()),
            }
            history.append({"agent_id": agent_id, "text": text})
            round_turns.append(turn)

        # Compute similarity between agents this round
        sim = SequenceMatcher(
            None, round_turns[0]["text"], round_turns[1]["text"]
        ).ratio()

        rounds_data.append({
            "round": r,
            "agent_0": round_turns[0],
            "agent_1": round_turns[1],
            "similarity": round(sim, 4),
            "either_degenerate": round_turns[0]["is_degenerate"] or round_turns[1]["is_degenerate"],
        })

        status = "DEGEN" if rounds_data[-1]["either_degenerate"] else "ok"
        print(f"  R{r}: sim={sim:.3f} top_freq=({round_turns[0]['top_word_freq']:.3f}, "
              f"{round_turns[1]['top_word_freq']:.3f}) [{status}]")

    # Summary
    degen_rounds = [r for r in rounds_data if r["either_degenerate"]]
    first_degen = degen_rounds[0]["round"] if degen_rounds else None

    return {
        "task": task_name,
        "condition": condition,
        "n_rounds": n_rounds,
        "rounds": rounds_data,
        "degeneration_count": len(degen_rounds),
        "first_degeneration_round": first_degen,
        "total_degenerate_fraction": len(degen_rounds) / n_rounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-3-7B-Instruct")
    parser.add_argument("--output_dir", default="./ablation_context_truncation")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_model(args.model)

    all_results = []

    for task_name, task_prompt in TASKS.items():
        for condition in CONTEXT_CONDITIONS:
            for rep in range(args.reps):
                print(f"\n{'='*60}")
                print(f"Task: {task_name} | Condition: {condition} | Rep: {rep}")
                print(f"{'='*60}")

                result = run_ablation(
                    model, tokenizer, task_name, task_prompt,
                    condition, n_rounds=args.rounds
                )
                result["rep"] = rep

                # Save individual trace
                fname = f"{task_name}_{condition}_rep{rep}.json"
                with open(os.path.join(args.output_dir, fname), "w") as f:
                    json.dump(result, f, indent=2)

                all_results.append({
                    "task": task_name,
                    "condition": condition,
                    "rep": rep,
                    "degeneration_count": result["degeneration_count"],
                    "first_degeneration_round": result["first_degeneration_round"],
                    "total_degenerate_fraction": result["total_degenerate_fraction"],
                })

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<12} {'Task':<20} {'Degen Rate':<12} {'First Degen':<12}")
    print("-" * 56)

    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)

    for condition in CONTEXT_CONDITIONS:
        results = by_condition[condition]
        degen_rate = sum(1 for r in results if r["degeneration_count"] > 0) / len(results)
        avg_first = [r["first_degeneration_round"] for r in results
                     if r["first_degeneration_round"] is not None]
        avg_first_str = f"{sum(avg_first)/len(avg_first):.1f}" if avg_first else "never"
        print(f"{condition:<12} {'(all tasks)':<20} {degen_rate:.0%}{'':>8} {avg_first_str}")

    # Save summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
