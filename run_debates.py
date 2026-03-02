#!/usr/bin/env python3
"""
run_debates.py — Multi-Agent Debate Orchestrator for OLMo 2

Runs structured multi-agent brainstorming sessions using OLMo 2 instances
with different system prompts and decoding parameters. Extracts hidden states
at each turn for H5 (representational convergence) analysis.

Outputs debate traces in the standard JSON format consumed by
measure_entropy.py and pid_analysis.py.

Communication conditions:
- natural_sequential: agents respond in fixed order, seeing all prior turns
- randomized_sequential: turn order randomized each round
- simultaneous_reveal: agents generate independently, then see all outputs

Author: Chaewon (Paper A collaboration with Herald)
"""

import argparse
import json
import os
import time
import random
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# ============================================================
# TASK BATTERY
# ============================================================
TASKS = {
    "startup_ideation_1": {
        "type": "startup_ideation",
        "prompt": "Propose a viable startup that addresses problems in online education. "
                  "Describe the core product, target market, and business model.",
    },
    "startup_ideation_2": {
        "type": "startup_ideation",
        "prompt": "Propose a viable startup that addresses problems in urban food waste. "
                  "Describe the core product, target market, and business model.",
    },
    "policy_design_1": {
        "type": "policy_design",
        "prompt": "Design a policy to address the growing problem of algorithmic hiring bias. "
                  "Consider stakeholders, enforcement mechanisms, and tradeoffs.",
    },
    "policy_design_2": {
        "type": "policy_design",
        "prompt": "Design a policy to address the mental health impact of social media on teenagers. "
                  "Consider stakeholders, enforcement mechanisms, and tradeoffs.",
    },
    "research_question_1": {
        "type": "research_question",
        "prompt": "Identify the most important open research question in interpretable machine learning. "
                  "Explain why it matters and what a solution would look like.",
    },
    "research_question_2": {
        "type": "research_question",
        "prompt": "Identify the most important open research question in synthetic biology. "
                  "Explain why it matters and what a solution would look like.",
    },
    "creative_problem_1": {
        "type": "creative_problem",
        "prompt": "A city has 1 million people but can only build 3 new buildings. "
                  "What should they build to maximize quality of life? Justify your choices.",
    },
    "creative_problem_2": {
        "type": "creative_problem",
        "prompt": "Design a school curriculum for a subject that doesn't exist yet but should. "
                  "What is the subject, why does it matter, and what would students learn?",
    },
    "ethical_dilemma_1": {
        "type": "ethical_dilemma",
        "prompt": "An AI system can predict criminal recidivism with 85% accuracy. "
                  "Should it be used in sentencing decisions? Argue your position.",
    },
    "ethical_dilemma_2": {
        "type": "ethical_dilemma",
        "prompt": "A pharmaceutical company develops a life-saving drug but prices it at $100,000/year. "
                  "What is the right balance between innovation incentives and access? Argue your position.",
    },
}

# ============================================================
# AGENT CONFIGURATIONS
# ============================================================
AGENT_CONFIGS = {
    "homogeneous": [
        {"agent_id": "agent_0", "system_prompt": "You are a helpful assistant participating in a brainstorming discussion. Share your ideas clearly and build on others' suggestions.", "temperature": 0.7, "top_k": 50},
        {"agent_id": "agent_1", "system_prompt": "You are a helpful assistant participating in a brainstorming discussion. Share your ideas clearly and build on others' suggestions.", "temperature": 0.7, "top_k": 50},
    ],
    "prompt_heterogeneous": [
        {"agent_id": "agent_0", "system_prompt": "You are a bold, creative thinker who prioritizes novel and unconventional ideas. Challenge assumptions and propose surprising solutions. Be specific and concrete.", "temperature": 0.7, "top_k": 50},
        {"agent_id": "agent_1", "system_prompt": "You are a careful, analytical thinker who prioritizes feasibility and evidence. Identify potential problems with proposals and suggest practical improvements. Be specific and concrete.", "temperature": 0.7, "top_k": 50},
    ],
    "config_heterogeneous": [
        {"agent_id": "agent_0", "system_prompt": "You are a helpful assistant participating in a brainstorming discussion. Share your ideas clearly and build on others' suggestions.", "temperature": 0.5, "top_k": 30},
        {"agent_id": "agent_1", "system_prompt": "You are a helpful assistant participating in a brainstorming discussion. Share your ideas clearly and build on others' suggestions.", "temperature": 1.0, "top_k": 80},
    ],
    "fully_heterogeneous": [
        {"agent_id": "agent_0", "system_prompt": "You are a bold, creative thinker who prioritizes novel and unconventional ideas. Challenge assumptions and propose surprising solutions. Be specific and concrete.", "temperature": 0.5, "top_k": 30},
        {"agent_id": "agent_1", "system_prompt": "You are a careful, analytical thinker who prioritizes feasibility and evidence. Identify potential problems with proposals and suggest practical improvements. Be specific and concrete.", "temperature": 1.0, "top_k": 80},
    ],
}

COMMUNICATION_CONDITIONS = ["natural_sequential", "randomized_sequential", "simultaneous_reveal"]


# ============================================================
# MODEL LOADING
# ============================================================
def load_models(model_name: str = "allenai/OLMo-2-0425-1B", embed_model: str = "all-MiniLM-L6-v2"):
    """Load OLMo 2 and sentence embedding model."""
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda",
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading embedding model {embed_model}...")
    embedder = SentenceTransformer(embed_model)

    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer, embedder


# ============================================================
# GENERATION WITH HIDDEN STATE EXTRACTION
# ============================================================
def generate_turn(
    model, tokenizer, prompt: str, system_prompt: str,
    temperature: float = 0.7, top_k: int = 50, max_new_tokens: int = 200
) -> tuple[str, list[float], float]:
    """
    Generate a single turn and extract final-layer hidden state.

    Returns: (generated_text, hidden_state_mean, generation_time_ms)
    """
    # Build chat-style prompt
    full_prompt = f"System: {system_prompt}\n\n{prompt}\n\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    gen_time_ms = (time.time() - start_time) * 1000

    # Decode generated text (excluding prompt)
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Extract final-layer hidden state from the last generated token
    # outputs.hidden_states is a tuple of (num_generated_tokens,) where each is
    # a tuple of (num_layers,) tensors of shape (batch, seq_len, hidden_dim)
    # We want the final layer's hidden state averaged across generated tokens
    try:
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # For each generated token step, get the last layer's hidden state
            final_layer_states = []
            for step_hidden in outputs.hidden_states:
                # step_hidden is tuple of (num_layers,) tensors
                # Last layer: step_hidden[-1] shape (batch, seq_len, hidden_dim)
                last_layer = step_hidden[-1]  # (batch, seq_len, hidden_dim)
                # Take the last token's hidden state
                final_layer_states.append(last_layer[0, -1, :].cpu().numpy())

            # Average across all generated tokens
            hidden_state_mean = np.mean(final_layer_states, axis=0).tolist()
        else:
            hidden_state_mean = []
    except Exception as e:
        print(f"  Warning: hidden state extraction failed: {e}")
        hidden_state_mean = []

    return generated_text, hidden_state_mean, gen_time_ms


# ============================================================
# DEBATE SESSION
# ============================================================
def run_debate_session(
    model, tokenizer, embedder,
    task_id: str, task_info: dict,
    agent_configs: list[dict], config_name: str,
    communication_condition: str,
    n_rounds: int = 10,
    max_tokens_per_turn: int = 200,
) -> dict:
    """Run a single multi-agent debate session."""

    session_id = f"{task_id}_{config_name}_{communication_condition}"
    print(f"\n{'='*60}")
    print(f"SESSION: {session_id}")
    print(f"Task: {task_info['prompt'][:80]}...")
    print(f"Agents: {config_name} | Condition: {communication_condition}")
    print(f"{'='*60}")

    conversation_history = []  # list of (agent_id, text) tuples
    rounds = []
    base_ts = int(time.time() * 1000)

    for r in range(n_rounds):
        print(f"\n  Round {r}...")
        round_turns = []

        # Determine turn order
        agents = list(agent_configs)
        if communication_condition == "randomized_sequential":
            random.shuffle(agents)

        if communication_condition == "simultaneous_reveal":
            # All agents generate independently (don't see each other's current-round output)
            turn_results = []
            for agent in agents:
                # Build prompt with history up to previous round only
                prompt = _build_prompt(task_info["prompt"], conversation_history, agent["agent_id"])
                text, hidden_state, gen_time = generate_turn(
                    model, tokenizer, prompt, agent["system_prompt"],
                    agent["temperature"], agent["top_k"], max_tokens_per_turn
                )
                embedding = embedder.encode(text).tolist()
                ts = base_ts + r * 30000  # same timestamp for simultaneous
                turn_results.append((agent, text, hidden_state, gen_time, embedding, ts))

            # Now add all turns to history simultaneously
            for agent, text, hidden_state, gen_time, embedding, ts in turn_results:
                turn = {
                    "agent_id": agent["agent_id"],
                    "text": text,
                    "timestamp_ms": ts,
                    "generation_time_ms": round(gen_time, 1),
                    "embedding": embedding,
                    "hidden_states": {"final_layer_mean": hidden_state},
                }
                round_turns.append(turn)
                conversation_history.append((agent["agent_id"], text))
                print(f"    {agent['agent_id']}: {text[:80]}...")

        else:
            # Sequential: each agent sees previous agents' output in this round
            for i, agent in enumerate(agents):
                prompt = _build_prompt(task_info["prompt"], conversation_history, agent["agent_id"])
                text, hidden_state, gen_time = generate_turn(
                    model, tokenizer, prompt, agent["system_prompt"],
                    agent["temperature"], agent["top_k"], max_tokens_per_turn
                )
                embedding = embedder.encode(text).tolist()
                ts = base_ts + r * 30000 + i * random.randint(2000, 5000)

                turn = {
                    "agent_id": agent["agent_id"],
                    "text": text,
                    "timestamp_ms": ts,
                    "generation_time_ms": round(gen_time, 1),
                    "embedding": embedding,
                    "hidden_states": {"final_layer_mean": hidden_state},
                }
                round_turns.append(turn)
                conversation_history.append((agent["agent_id"], text))
                print(f"    {agent['agent_id']}: {text[:80]}...")

        rounds.append({"round": r, "turns": round_turns})

    trace = {
        "session_id": session_id,
        "task_type": task_info["type"],
        "task_prompt": task_info["prompt"],
        "communication_condition": communication_condition,
        "agent_configs": agent_configs,
        "rounds": rounds,
        "metadata": {
            "model": "allenai/OLMo-2-0425-1B",
            "num_rounds": n_rounds,
            "max_tokens_per_turn": max_tokens_per_turn,
            "date": time.strftime("%Y-%m-%d"),
        }
    }
    return trace


def _build_prompt(task_prompt: str, history: list[tuple[str, str]], current_agent: str) -> str:
    """Build the conversation prompt for an agent, including task and history."""
    parts = [f"Topic: {task_prompt}"]

    if history:
        parts.append("\nDiscussion so far:")
        for agent_id, text in history:
            label = "You" if agent_id == current_agent else agent_id
            parts.append(f"\n{label}: {text}")

    parts.append(f"\nNow it's your turn. Share your perspective on this topic. "
                 f"Build on or respectfully disagree with what others have said. "
                 f"Be specific and concise (2-3 paragraphs).")

    return "\n".join(parts)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run multi-agent debates with OLMo 2")
    parser.add_argument("--output_dir", default="./debate_traces")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--n_rounds", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # Pilot mode: run a subset
    parser.add_argument("--pilot", action="store_true", help="Run 6 pilot sessions only")

    # Or specify exact tasks/configs/conditions
    parser.add_argument("--tasks", nargs="+", default=None, help="Task IDs to run")
    parser.add_argument("--configs", nargs="+", default=None, help="Agent config names")
    parser.add_argument("--conditions", nargs="+", default=None, help="Communication conditions")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load models
    model, tokenizer, embedder = load_models(args.model, args.embed_model)

    # Determine which sessions to run
    if args.pilot:
        # Pilot: 2 tasks × 1 config × 3 conditions = 6 sessions
        task_ids = ["startup_ideation_1", "policy_design_1"]
        config_names = ["prompt_heterogeneous"]
        conditions = COMMUNICATION_CONDITIONS
    else:
        task_ids = args.tasks or list(TASKS.keys())
        config_names = args.configs or list(AGENT_CONFIGS.keys())
        conditions = args.conditions or COMMUNICATION_CONDITIONS

    total = len(task_ids) * len(config_names) * len(conditions)
    print(f"\n🔬 Running {total} debate sessions")
    print(f"   Tasks: {task_ids}")
    print(f"   Configs: {config_names}")
    print(f"   Conditions: {conditions}")
    print(f"   Rounds per session: {args.n_rounds}")

    session_count = 0
    for task_id in task_ids:
        for config_name in config_names:
            for condition in conditions:
                session_count += 1
                print(f"\n[{session_count}/{total}]")

                trace = run_debate_session(
                    model, tokenizer, embedder,
                    task_id, TASKS[task_id],
                    AGENT_CONFIGS[config_name], config_name,
                    condition,
                    n_rounds=args.n_rounds,
                    max_tokens_per_turn=args.max_tokens,
                )

                # Save trace
                out_path = os.path.join(args.output_dir, f"{trace['session_id']}.json")
                with open(out_path, 'w') as f:
                    json.dump(trace, f, indent=2)
                print(f"\n  ✅ Saved: {out_path}")

    print(f"\n🔬 All {total} sessions complete. Traces in {args.output_dir}")
    print(f"   Next: run measure_entropy.py and pid_analysis.py on these traces")


if __name__ == "__main__":
    main()
