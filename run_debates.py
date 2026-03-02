#!/usr/bin/env python3
"""
Run multi-agent debate sessions using OLMo 2 (7B) locally on GPU.
Saves full conversation traces for downstream analysis.
"""
import json
import os
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Task battery
TASKS = {
    "startup_1": "Propose a YC-viable startup that addresses the inefficiency of hiring processes for technical roles. Explain your idea, target market, business model, and why now is the right time.",
    "startup_2": "Propose a YC-viable startup that addresses mental health support for remote workers. Explain your idea, target market, business model, and why now is the right time.",
    "policy_1": "Design a policy to address the growing problem of AI-generated misinformation in elections. Consider tradeoffs between free speech, safety, and enforcement feasibility.",
    "policy_2": "Design a policy to address housing affordability in major US cities. Consider tradeoffs between development, community impact, and economic incentives.",
    "research_1": "What is the most important open question in the field of multi-agent AI systems? Justify why this question matters more than alternatives.",
    "research_2": "What is the most important open question in AI alignment research? Justify why this question matters more than alternatives.",
    "creative_1": "How would you design a city for 100,000 people on Mars with only resources that can be manufactured locally? Propose a specific plan.",
    "creative_2": "How would you redesign the education system from scratch if students had unlimited access to AI tutors? Propose a specific plan.",
    "ethical_1": "A self-driving car must choose between swerving to avoid a pedestrian (risking the passenger) or staying course (risking the pedestrian). The pedestrian is jaywalking. What should the car do and why?",
    "ethical_2": "A hospital has one organ available for transplant. Patient A is 25 with a family and high survival odds. Patient B is 60, a renowned scientist working on a cancer cure, with moderate survival odds. Who should receive the organ and why?"
}

SYSTEM_PROMPTS = {
    "agent_a": "You are a thoughtful analytical thinker. You approach problems systematically, considering evidence and logic. You are willing to change your mind when presented with good arguments, but you don't cave to social pressure alone. Engage substantively with other perspectives.",
    "agent_b": "You are a creative lateral thinker. You approach problems by questioning assumptions and exploring unconventional angles. You value novelty and originality. You are willing to find common ground but resist premature consensus. Engage substantively with other perspectives."
}

NUM_ROUNDS = 8


def load_model(model_name="allenai/OLMo-2-0425-1B"):
    """Load OLMo 2 model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded. Device: {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate a response given conversation history."""
    # Build prompt from messages
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"System: {content}\n\n"
        elif role == "user":
            prompt += f"Other agent: {content}\n\n"
        elif role == "assistant":
            prompt += f"You: {content}\n\n"
    prompt += "You:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Truncate at common stop patterns
    for stop in ["\nOther agent:", "\nSystem:", "\nYou:"]:
        if stop in response:
            response = response[:response.index(stop)].strip()

    return response


def run_debate_sequential(model, tokenizer, task_id, task_prompt, condition="natural"):
    """Run a single multi-agent debate session."""
    trace = {
        "task_id": task_id,
        "task_prompt": task_prompt,
        "condition": condition,
        "num_rounds": NUM_ROUNDS,
        "system_prompts": SYSTEM_PROMPTS,
        "started_at": datetime.utcnow().isoformat(),
        "rounds": []
    }

    history_a = [{"role": "system", "content": SYSTEM_PROMPTS["agent_a"]}]
    history_b = [{"role": "system", "content": SYSTEM_PROMPTS["agent_b"]}]

    # Initial prompt
    initial = f"Topic for discussion: {task_prompt}\n\nPlease share your initial thoughts and proposal. Be specific and substantive."
    history_a.append({"role": "user", "content": initial})
    history_b.append({"role": "user", "content": initial})

    for round_num in range(NUM_ROUNDS):
        round_data = {"round": round_num, "turns": []}

        if condition == "natural":
            # Agent A goes first, then B
            order = ["agent_a", "agent_b"]
        elif condition == "randomized":
            order = random.sample(["agent_a", "agent_b"], 2)
        elif condition == "simultaneous":
            order = ["simultaneous"]
        else:
            order = ["agent_a", "agent_b"]

        if condition == "simultaneous":
            # Both generate without seeing the other's response
            t0 = time.time()
            resp_a = generate_response(model, tokenizer, history_a)
            t_a = time.time() - t0

            t0 = time.time()
            resp_b = generate_response(model, tokenizer, history_b)
            t_b = time.time() - t0

            round_data["turns"].append({
                "agent": "agent_a", "response": resp_a,
                "latency_sec": t_a, "timestamp": datetime.utcnow().isoformat()
            })
            round_data["turns"].append({
                "agent": "agent_b", "response": resp_b,
                "latency_sec": t_b, "timestamp": datetime.utcnow().isoformat()
            })

            # Now reveal to each other
            history_a.append({"role": "assistant", "content": resp_a})
            history_a.append({"role": "user", "content": f"[Simultaneous reveal] The other agent said: {resp_b}\n\nRespond to their points and develop your position further."})
            history_b.append({"role": "assistant", "content": resp_b})
            history_b.append({"role": "user", "content": f"[Simultaneous reveal] The other agent said: {resp_a}\n\nRespond to their points and develop your position further."})
        else:
            for agent_id in order:
                if agent_id == "agent_a":
                    t0 = time.time()
                    resp = generate_response(model, tokenizer, history_a)
                    latency = time.time() - t0
                    history_a.append({"role": "assistant", "content": resp})
                    history_b.append({"role": "user", "content": resp})
                else:
                    t0 = time.time()
                    resp = generate_response(model, tokenizer, history_b)
                    latency = time.time() - t0
                    history_b.append({"role": "assistant", "content": resp})
                    history_a.append({"role": "user", "content": resp})

                round_data["turns"].append({
                    "agent": agent_id, "response": resp,
                    "latency_sec": latency, "timestamp": datetime.utcnow().isoformat()
                })

        trace["rounds"].append(round_data)
        print(f"  Round {round_num + 1}/{NUM_ROUNDS} complete")

    trace["finished_at"] = datetime.utcnow().isoformat()
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B", help="Model name")
    parser.add_argument("--output-dir", default="./traces", help="Output directory")
    parser.add_argument("--conditions", nargs="+", default=["natural", "randomized", "simultaneous"])
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific task IDs (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_model(args.model)

    task_ids = args.tasks or list(TASKS.keys())

    total = len(task_ids) * len(args.conditions)
    done = 0

    for condition in args.conditions:
        for task_id in task_ids:
            done += 1
            print(f"\n[{done}/{total}] Task: {task_id} | Condition: {condition}")
            trace = run_debate_sequential(model, tokenizer, task_id, TASKS[task_id], condition)

            outpath = Path(args.output_dir) / f"{task_id}_{condition}.json"
            with open(outpath, "w") as f:
                json.dump(trace, f, indent=2)
            print(f"  Saved: {outpath}")

    print(f"\nDone! {done} sessions saved to {args.output_dir}")


if __name__ == "__main__":
    main()
