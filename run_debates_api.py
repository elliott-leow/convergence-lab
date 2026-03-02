#!/usr/bin/env python3
"""
run_debates_api.py — Multi-Agent Debate via API (OpenAI / Anthropic)
Cross-model validation for the Divergence Default paper.

Runs the same debate structure as run_debates.py but against cloud APIs
instead of local inference. Outputs traces in identical JSON format so
all analysis scripts (measure_entropy, pid_analysis_v3, analyze) work
unchanged.

Usage:
    # OpenAI (GPT-4o-mini)
    export OPENAI_API_KEY=sk-...
    python run_debates_api.py --provider openai --model gpt-4o-mini \
        --sessions 6 --output_dir ./traces_gpt4omini

    # Anthropic (Claude Haiku)
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_debates_api.py --provider anthropic --model claude-3-5-haiku-20241022 \
        --sessions 6 --output_dir ./traces_haiku

    # Full cross-model comparison (hetero + homo × 2 tasks × 1 condition)
    python run_debates_api.py --provider openai --model gpt-4o-mini \
        --sessions 12 --output_dir ./traces_gpt4omini --include_homo

Cost estimate: ~$2-5 for 12 sessions at 10 rounds each with mini/haiku models.

Author: Herald (Paper A collaboration with Chaewon)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# API clients — import what's available
# ---------------------------------------------------------------------------
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("WARNING: sentence-transformers not installed. Traces will lack embeddings.")
    print("Run measure_entropy.py with --embed to add them later.")

# ---------------------------------------------------------------------------
# Task definitions (same as run_debates.py)
# ---------------------------------------------------------------------------
TASKS = {
    "startup_ideation": {
        "prompt": "Propose an innovative startup idea that addresses a real problem. "
                  "Describe the problem, your solution, target market, and why now is the right time.",
        "type": "open_ended",
    },
    "policy_design": {
        "prompt": "Design a policy to address the housing affordability crisis in major US cities. "
                  "Consider economic incentives, zoning reform, tenant protections, and funding mechanisms.",
        "type": "constrained",
    },
    "ethical_dilemma": {
        "prompt": "A self-driving car must choose between swerving to avoid a pedestrian (risking "
                  "the passenger) or staying course (risking the pedestrian). How should the AI be "
                  "programmed to handle this, and who bears moral responsibility?",
        "type": "constrained",
    },
    "research_question": {
        "prompt": "Propose a novel research question at the intersection of AI and neuroscience. "
                  "Describe the hypothesis, methodology, expected findings, and potential impact.",
        "type": "open_ended",
    },
    "creative_problem": {
        "prompt": "How might we redesign public transportation for a post-pandemic world where "
                  "remote work is common but cities still need to function? Think beyond buses and trains.",
        "type": "open_ended",
    },
}

# ---------------------------------------------------------------------------
# Agent personas
# ---------------------------------------------------------------------------
PERSONAS = {
    "heterogeneous": {
        "agent_a": {
            "system_prompt": (
                "You are an analytical, systems-thinking debater. You approach problems "
                "through data, evidence, and logical frameworks. You value precision, "
                "measurability, and second-order effects. When you disagree, you cite "
                "specific reasoning. Keep responses focused and substantive (2-4 paragraphs)."
            ),
            "temperature": 0.7,
        },
        "agent_b": {
            "system_prompt": (
                "You are a creative, human-centered debater. You approach problems through "
                "empathy, narrative, and lateral thinking. You value user experience, equity, "
                "and unintended consequences on real people. When you disagree, you tell "
                "stories and use analogies. Keep responses focused and substantive (2-4 paragraphs)."
            ),
            "temperature": 0.9,
        },
    },
    "homogeneous": {
        "agent_a": {
            "system_prompt": (
                "You are a helpful assistant participating in a collaborative discussion. "
                "Share your analysis and recommendations on the topic. "
                "Keep responses focused and substantive (2-4 paragraphs)."
            ),
            "temperature": 0.7,
        },
        "agent_b": {
            "system_prompt": (
                "You are a helpful assistant participating in a collaborative discussion. "
                "Share your analysis and recommendations on the topic. "
                "Keep responses focused and substantive (2-4 paragraphs)."
            ),
            "temperature": 0.7,
        },
    },
}


# ---------------------------------------------------------------------------
# API wrappers
# ---------------------------------------------------------------------------

def call_openai(model: str, system_prompt: str, messages: list[dict],
                temperature: float = 0.7) -> str:
    """Call OpenAI-compatible API."""
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed. pip install openai")

    client = openai.OpenAI()
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def call_anthropic(model: str, system_prompt: str, messages: list[dict],
                   temperature: float = 0.7) -> str:
    """Call Anthropic API."""
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed. pip install anthropic")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        temperature=temperature,
    )
    return response.content[0].text


def call_api(provider: str, model: str, system_prompt: str,
             messages: list[dict], temperature: float = 0.7) -> str:
    """Unified API caller."""
    if provider == "openai":
        return call_openai(model, system_prompt, messages, temperature)
    elif provider == "anthropic":
        return call_anthropic(model, system_prompt, messages, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Debate runner
# ---------------------------------------------------------------------------

def run_debate_session(
    provider: str,
    model: str,
    task_name: str,
    task_prompt: str,
    agent_a_config: dict,
    agent_b_config: dict,
    persona_type: str,
    n_rounds: int = 10,
    communication: str = "natural_sequential",
    embedder=None,
) -> dict:
    """
    Run a single debate session between two API-backed agents.

    Returns trace dict in the same format as run_debates.py.
    """
    session_id = f"{task_name}_{persona_type}_{communication}_{int(time.time())}"
    print(f"\n--- Session: {session_id} ---")

    # Conversation histories (what each agent sees)
    agent_a_history = []
    agent_b_history = []

    rounds = []

    for r in range(n_rounds):
        print(f"  Round {r+1}/{n_rounds}...", end=" ", flush=True)
        round_data = {"round": r, "turns": []}

        # Determine turn order
        if communication == "randomized_sequential":
            import random
            order = random.sample(["a", "b"], 2)
        else:
            order = ["a", "b"]

        for agent_id in order:
            if agent_id == "a":
                config = agent_a_config
                history = agent_a_history
                label = "agent_a"
            else:
                config = agent_b_config
                history = agent_b_history
                label = "agent_b"

            # Build message context
            if r == 0 and len(history) == 0:
                # First turn: just the task prompt
                messages = [{"role": "user", "content": f"Topic for discussion:\n\n{task_prompt}"}]
            else:
                messages = list(history)
                if communication == "simultaneous" and r > 0:
                    # In simultaneous mode, both agents see each other's last response at the same time
                    pass  # history already updated after both spoke
                # Add a nudge for continuation
                if messages and messages[-1]["role"] == "assistant":
                    messages.append({
                        "role": "user",
                        "content": "Please continue the discussion. Build on, challenge, or extend the previous points."
                    })

            # Call API
            try:
                response_text = call_api(
                    provider, model,
                    config["system_prompt"],
                    messages,
                    config["temperature"],
                )
            except Exception as e:
                print(f"API error: {e}")
                response_text = f"[API ERROR: {e}]"

            # Compute embedding if available
            embedding = None
            if embedder is not None:
                embedding = embedder.encode(response_text).tolist()

            turn = {
                "agent_id": label,
                "text": response_text,
                "embedding": embedding,
                "round": r,
            }
            round_data["turns"].append(turn)

            # Update BOTH agents' histories (they see each other's responses)
            msg_from_agent = {"role": "assistant", "content": response_text}
            msg_as_other = {"role": "user", "content": f"[{label}]: {response_text}"}

            if agent_id == "a":
                agent_a_history.append(msg_from_agent)
                agent_b_history.append(msg_as_other)
            else:
                agent_b_history.append(msg_from_agent)
                agent_a_history.append(msg_as_other)

        rounds.append(round_data)
        print("done", flush=True)

    trace = {
        "session_id": session_id,
        "model": model,
        "provider": provider,
        "task_type": task_name,
        "task_prompt": task_prompt,
        "persona_type": persona_type,
        "communication_condition": communication,
        "n_rounds": n_rounds,
        "agent_configs": [
            {"agent_id": "agent_a", "system_prompt": agent_a_config["system_prompt"],
             "temperature": agent_a_config["temperature"]},
            {"agent_id": "agent_b", "system_prompt": agent_b_config["system_prompt"],
             "temperature": agent_b_config["temperature"]},
        ],
        "rounds": rounds,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="API-based multi-agent debate runner")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic"],
                        help="API provider")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., gpt-4o-mini, claude-3-5-haiku-20241022)")
    parser.add_argument("--output_dir", default="./traces_api",
                        help="Output directory for trace JSON files")
    parser.add_argument("--sessions", type=int, default=6,
                        help="Total number of sessions to run")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Rounds per session")
    parser.add_argument("--include_homo", action="store_true",
                        help="Include homogeneous persona condition (doubles session count)")
    parser.add_argument("--tasks", nargs="+", default=["startup_ideation", "policy_design"],
                        help="Task names to use")
    parser.add_argument("--communication", default="natural_sequential",
                        choices=["natural_sequential", "randomized_sequential", "simultaneous"],
                        help="Communication condition")
    parser.add_argument("--embed", action="store_true",
                        help="Compute embeddings during generation (requires sentence-transformers)")
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2",
                        help="Embedding model name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate API access
    if args.provider == "openai" and not HAS_OPENAI:
        print("ERROR: pip install openai")
        sys.exit(1)
    if args.provider == "anthropic" and not HAS_ANTHROPIC:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    # Check API key
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Load embedder
    embedder = None
    if args.embed and HAS_ST:
        print(f"Loading embedding model: {args.embed_model}")
        embedder = SentenceTransformer(args.embed_model)

    # Build session plan
    persona_types = ["heterogeneous"]
    if args.include_homo:
        persona_types.append("homogeneous")

    task_names = [t for t in args.tasks if t in TASKS]
    if not task_names:
        print(f"ERROR: No valid tasks. Choose from: {list(TASKS.keys())}")
        sys.exit(1)

    # Distribute sessions across tasks and personas
    sessions_per_cell = max(1, args.sessions // (len(task_names) * len(persona_types)))
    total_planned = sessions_per_cell * len(task_names) * len(persona_types)

    print(f"\n{'='*60}")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Tasks: {task_names}")
    print(f"  Personas: {persona_types}")
    print(f"  Sessions per cell: {sessions_per_cell}")
    print(f"  Total sessions: {total_planned}")
    print(f"  Rounds per session: {args.rounds}")
    print(f"  Communication: {args.communication}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    session_count = 0
    for persona_type in persona_types:
        agents = PERSONAS[persona_type]
        for task_name in task_names:
            task = TASKS[task_name]
            for rep in range(sessions_per_cell):
                session_count += 1
                print(f"\n[{session_count}/{total_planned}] "
                      f"{task_name} / {persona_type} / rep {rep+1}")

                trace = run_debate_session(
                    provider=args.provider,
                    model=args.model,
                    task_name=task_name,
                    task_prompt=task["prompt"],
                    agent_a_config=agents["agent_a"],
                    agent_b_config=agents["agent_b"],
                    persona_type=persona_type,
                    n_rounds=args.rounds,
                    communication=args.communication,
                    embedder=embedder,
                )

                # Save trace
                out_file = os.path.join(
                    args.output_dir,
                    f"{trace['session_id']}.json"
                )
                with open(out_file, 'w') as f:
                    json.dump(trace, f, indent=2)
                print(f"  Saved: {out_file}")

                # Brief delay to avoid rate limits
                time.sleep(1)

    print(f"\n{'='*60}")
    print(f"  Complete! {session_count} sessions saved to {args.output_dir}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Add embeddings (if not done): python measure_entropy.py --traces_dir {args.output_dir} --embed")
    print(f"  2. Run entropy analysis: python measure_entropy.py --traces_dir {args.output_dir}")
    print(f"  3. Run coupling analysis: python pid_analysis_v3.py --traces_dir {args.output_dir} --plot")
    print(f"  4. Compare with OLMo results!")


if __name__ == "__main__":
    main()
