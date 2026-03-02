#!/usr/bin/env python3
"""
Pre-debate self-consistency measurement (H3).
Run each agent 10x solo on the same prompt, compute variance.
Tests inverted-U hypothesis: moderate consistency → best debate partner.
"""
import json
import argparse
import numpy as np
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist


SOLO_PROMPTS = {
    "startup": "Propose a YC-viable startup. Explain your idea, target market, business model, and why now is the right time.",
    "policy": "Design a policy to address AI-generated misinformation. Consider tradeoffs between free speech, safety, and enforcement.",
    "research": "What is the most important open question in multi-agent AI systems? Justify your answer.",
    "creative": "How would you design a city for 100,000 people on Mars? Propose a specific plan.",
    "ethical": "A self-driving car must choose between swerving (risking passenger) or staying course (risking pedestrian). What should it do and why?"
}

SYSTEM_PROMPTS = {
    "agent_a": "You are a thoughtful analytical thinker. You approach problems systematically, considering evidence and logic.",
    "agent_b": "You are a creative lateral thinker. You approach problems by questioning assumptions and exploring unconventional angles."
}

NUM_SOLO_RUNS = 10


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_prompt, max_new_tokens=512):
    prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_self_consistency(responses, embedder):
    """
    Compute self-consistency as 1 - mean pairwise cosine distance.
    High value = very consistent (always says same thing).
    Low value = high variance (says different things each time).
    """
    embeddings = embedder.encode(responses, show_progress_bar=False)
    distances = pdist(embeddings, metric="cosine")
    mean_distance = np.mean(distances)
    return {
        "consistency_score": float(1 - mean_distance),
        "mean_pairwise_distance": float(mean_distance),
        "std_pairwise_distance": float(np.std(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--output", default="./analysis/self_consistency.json")
    parser.add_argument("--num-runs", type=int, default=NUM_SOLO_RUNS)
    args = parser.parse_args()

    import os
    os.makedirs(Path(args.output).parent, exist_ok=True)

    model, tokenizer = load_model(args.model)
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    results = {}

    for agent_id, sys_prompt in SYSTEM_PROMPTS.items():
        results[agent_id] = {}
        for task_id, task_prompt in SOLO_PROMPTS.items():
            print(f"\n{agent_id} / {task_id}: generating {args.num_runs} solo responses...")
            responses = []
            for i in range(args.num_runs):
                resp = generate(model, tokenizer, sys_prompt, task_prompt)
                responses.append(resp)
                print(f"  Run {i+1}/{args.num_runs} done ({len(resp)} chars)")

            consistency = compute_self_consistency(responses, embedder)
            results[agent_id][task_id] = {
                "responses": responses,
                "consistency": consistency
            }
            print(f"  Consistency score: {consistency['consistency_score']:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
