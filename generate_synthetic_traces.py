#!/usr/bin/env python3
"""
generate_synthetic_traces.py — Synthetic debate traces for pipeline validation

Generates two test scenarios with known ground-truth dynamics:

1. CONVERGING: Agents start in different semantic clusters, gradually merge.
   Expected PID: rising redundancy, falling synergy over rounds.
   Expected entropy: monotonically decreasing H(t), declining δ(t).

2. DIVERGING: Agents maintain distinct clusters throughout.
   Expected PID: stable/rising synergy, low redundancy.
   Expected entropy: stable H(t), stable δ(t).

Also generates:
3. SUDDEN_COLLAPSE: Agents diverge for 6 rounds then instantly converge.
   Expected: No critical slowing down — tests the cascade/tipping-point falsification (H2).

4. GRADUAL_CONVERGENCE: Slow convergence with δ(t) declining before H(t).
   Expected: Critical slowing down present — tests the phase-transition prediction (H2).

Usage:
    python generate_synthetic_traces.py --output_dir ./synthetic_traces --n_rounds 10
    python generate_synthetic_traces.py --output_dir ./synthetic_traces --scenario converging
    python generate_synthetic_traces.py --output_dir ./synthetic_traces --all

Author: Herald (Paper A collaboration with Chaewon)
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np


def make_embedding(cluster_center: np.ndarray, noise_scale: float = 0.05, dim: int = 384) -> list[float]:
    """Generate a noisy embedding near a cluster center."""
    noise = np.random.normal(0, noise_scale, dim)
    emb = cluster_center + noise
    # Normalize to unit sphere (sentence embeddings are typically normalized)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


def make_hidden_state(cluster_center: np.ndarray, noise_scale: float = 0.03, dim: int = 2048) -> list[float]:
    """Generate a synthetic hidden state (higher dim, different noise profile)."""
    noise = np.random.normal(0, noise_scale, dim)
    hs = cluster_center + noise
    return hs.tolist()


def make_turn(agent_id: str, text: str, embedding: list, hidden_state: list, timestamp_ms: int, gen_time_ms: int) -> dict:
    return {
        "agent_id": agent_id,
        "text": text,
        "timestamp_ms": timestamp_ms,
        "generation_time_ms": gen_time_ms,
        "embedding": embedding,
        "hidden_states": {
            "final_layer_mean": hidden_state
        }
    }


def generate_cluster_centers(n_clusters: int, dim: int, separation: float = 2.0) -> list[np.ndarray]:
    """Generate well-separated cluster centers in embedding space."""
    centers = []
    for i in range(n_clusters):
        center = np.random.normal(0, 1, dim)
        center = center / np.linalg.norm(center) * separation
        # Add offset to separate clusters
        offset = np.zeros(dim)
        offset[i % dim] = separation * (i + 1)
        center = center + offset
        center = center / np.linalg.norm(center)
        centers.append(center)
    return centers


# ============================================================
# SCENARIO 1: CONVERGING DEBATE
# ============================================================

def generate_converging(n_rounds: int = 10, seed: int = 42) -> dict:
    """
    Agents start at different cluster centers, gradually move toward a shared center.
    
    Agent 0 starts at center_A, Agent 1 starts at center_B.
    By round n_rounds, both are at center_merged = (center_A + center_B) / 2.
    Interpolation is linear — agent positions blend over rounds.
    
    Expected PID: redundancy rises, synergy falls.
    Expected H(t): monotonically decreasing.
    Expected δ(t): decreasing (each turn is closer to the previous).
    """
    np.random.seed(seed)
    random.seed(seed)

    emb_dim = 384
    hs_dim = 2048

    # Two distinct starting positions
    center_A = np.random.normal(0, 1, emb_dim)
    center_A = center_A / np.linalg.norm(center_A)
    center_B = np.random.normal(0, 1, emb_dim)
    center_B = center_B / np.linalg.norm(center_B)
    center_merged = (center_A + center_B) / 2
    center_merged = center_merged / np.linalg.norm(center_merged)

    # Hidden state centers (higher dim)
    hs_A = np.random.normal(0, 1, hs_dim)
    hs_B = np.random.normal(0, 1, hs_dim)
    hs_merged = (hs_A + hs_B) / 2

    rounds = []
    base_ts = 1709341200000

    for r in range(n_rounds):
        # Interpolation factor: 0 at round 0, 1 at final round
        alpha = r / max(n_rounds - 1, 1)

        # Agent 0 moves from center_A toward merged
        a0_center = (1 - alpha) * center_A + alpha * center_merged
        # Agent 1 moves from center_B toward merged
        a1_center = (1 - alpha) * center_B + alpha * center_merged

        # Hidden states follow same trajectory
        a0_hs_center = (1 - alpha) * hs_A + alpha * hs_merged
        a1_hs_center = (1 - alpha) * hs_B + alpha * hs_merged

        # Noise decreases as agents converge (they become more certain)
        noise = 0.08 * (1 - alpha * 0.7)

        ts = base_ts + r * 30000  # 30s between rounds
        turns = [
            make_turn(
                "agent_0",
                f"[CONVERGING r{r}] Agent 0 proposal (alpha={alpha:.2f})",
                make_embedding(a0_center, noise, emb_dim),
                make_hidden_state(a0_hs_center, noise * 0.5, hs_dim),
                ts,
                random.randint(2000, 4000)
            ),
            make_turn(
                "agent_1",
                f"[CONVERGING r{r}] Agent 1 proposal (alpha={alpha:.2f})",
                make_embedding(a1_center, noise, emb_dim),
                make_hidden_state(a1_hs_center, noise * 0.5, hs_dim),
                ts + random.randint(2000, 5000),
                random.randint(2000, 4000)
            ),
        ]

        rounds.append({"round": r, "turns": turns})

    return {
        "session_id": "synthetic_converging_01",
        "task_type": "synthetic_validation",
        "task_prompt": "Synthetic converging debate for pipeline validation",
        "communication_condition": "natural_sequential",
        "agent_configs": [
            {"agent_id": "agent_0", "system_prompt": "converging_A", "temperature": 0.7, "top_k": 50},
            {"agent_id": "agent_1", "system_prompt": "converging_B", "temperature": 0.7, "top_k": 50},
        ],
        "rounds": rounds,
        "metadata": {
            "model": "synthetic",
            "num_rounds": n_rounds,
            "scenario": "converging",
            "expected_pid": "rising redundancy, falling synergy",
            "expected_entropy": "monotonically decreasing H(t), declining δ(t)",
            "date": "2026-03-02"
        }
    }


# ============================================================
# SCENARIO 2: DIVERGING DEBATE
# ============================================================

def generate_diverging(n_rounds: int = 10, seed: int = 123) -> dict:
    """
    Agents maintain distinct positions throughout. Each explores its own region
    with some round-to-round variation but no convergence.
    
    Expected PID: stable/rising synergy, low redundancy.
    Expected H(t): stable (no compression).
    Expected δ(t): stable (agents keep producing novel content in their own region).
    """
    np.random.seed(seed)
    random.seed(seed)

    emb_dim = 384
    hs_dim = 2048

    center_A = np.random.normal(0, 1, emb_dim)
    center_A = center_A / np.linalg.norm(center_A)
    center_B = -center_A  # Maximally separated (opposite direction)

    hs_A = np.random.normal(0, 1, hs_dim)
    hs_B = np.random.normal(0, 1, hs_dim)

    rounds = []
    base_ts = 1709341200000

    for r in range(n_rounds):
        # Add random walk within each agent's region (exploration, no convergence)
        drift_A = np.random.normal(0, 0.02, emb_dim)
        drift_B = np.random.normal(0, 0.02, emb_dim)
        a0_center = center_A + drift_A * r  # Slow random walk
        a1_center = center_B + drift_B * r

        a0_hs = hs_A + np.random.normal(0, 0.02, hs_dim) * r
        a1_hs = hs_B + np.random.normal(0, 0.02, hs_dim) * r

        noise = 0.08  # Constant noise — no increasing certainty

        ts = base_ts + r * 30000
        turns = [
            make_turn(
                "agent_0",
                f"[DIVERGING r{r}] Agent 0 explores independently",
                make_embedding(a0_center, noise, emb_dim),
                make_hidden_state(a0_hs, noise * 0.5, hs_dim),
                ts,
                random.randint(2000, 4000)
            ),
            make_turn(
                "agent_1",
                f"[DIVERGING r{r}] Agent 1 explores independently",
                make_embedding(a1_center, noise, emb_dim),
                make_hidden_state(a1_hs, noise * 0.5, hs_dim),
                ts + random.randint(2000, 5000),
                random.randint(2000, 4000)
            ),
        ]

        rounds.append({"round": r, "turns": turns})

    return {
        "session_id": "synthetic_diverging_01",
        "task_type": "synthetic_validation",
        "task_prompt": "Synthetic diverging debate for pipeline validation",
        "communication_condition": "natural_sequential",
        "agent_configs": [
            {"agent_id": "agent_0", "system_prompt": "diverging_A", "temperature": 0.9, "top_k": 50},
            {"agent_id": "agent_1", "system_prompt": "diverging_B", "temperature": 0.9, "top_k": 50},
        ],
        "rounds": rounds,
        "metadata": {
            "model": "synthetic",
            "num_rounds": n_rounds,
            "scenario": "diverging",
            "expected_pid": "stable/rising synergy, low redundancy",
            "expected_entropy": "stable H(t), stable δ(t)",
            "date": "2026-03-02"
        }
    }


# ============================================================
# SCENARIO 3: SUDDEN COLLAPSE (H2 falsification test)
# ============================================================

def generate_sudden_collapse(n_rounds: int = 10, collapse_round: int = 7, seed: int = 456) -> dict:
    """
    Agents diverge freely for collapse_round rounds, then INSTANTLY converge.
    No gradual slowing down — a sudden cliff in entropy.
    
    Tests H2 falsification: if H(t) drops without preceding δ(t) decline,
    the dynamics are cascade/tipping-point, not phase-transition.
    
    Expected PID: sudden redundancy spike at collapse_round.
    Expected H(t): stable then cliff.
    Expected δ(t): stable then sudden drop (simultaneous with H, not preceding it).
    """
    np.random.seed(seed)
    random.seed(seed)

    emb_dim = 384
    hs_dim = 2048

    center_A = np.random.normal(0, 1, emb_dim)
    center_A = center_A / np.linalg.norm(center_A)
    center_B = np.random.normal(0, 1, emb_dim)
    center_B = center_B / np.linalg.norm(center_B)
    center_merged = (center_A + center_B) / 2
    center_merged = center_merged / np.linalg.norm(center_merged)

    hs_A = np.random.normal(0, 1, hs_dim)
    hs_B = np.random.normal(0, 1, hs_dim)
    hs_merged = (hs_A + hs_B) / 2

    rounds = []
    base_ts = 1709341200000

    for r in range(n_rounds):
        if r < collapse_round:
            # Pre-collapse: agents maintain distinct positions with normal variation
            a0_center = center_A + np.random.normal(0, 0.03, emb_dim)
            a1_center = center_B + np.random.normal(0, 0.03, emb_dim)
            a0_hs = hs_A + np.random.normal(0, 0.02, hs_dim)
            a1_hs = hs_B + np.random.normal(0, 0.02, hs_dim)
            noise = 0.08
        else:
            # Post-collapse: both agents snap to merged position
            a0_center = center_merged + np.random.normal(0, 0.01, emb_dim)
            a1_center = center_merged + np.random.normal(0, 0.01, emb_dim)
            a0_hs = hs_merged + np.random.normal(0, 0.01, hs_dim)
            a1_hs = hs_merged + np.random.normal(0, 0.01, hs_dim)
            noise = 0.02

        ts = base_ts + r * 30000
        turns = [
            make_turn(
                "agent_0",
                f"[SUDDEN_COLLAPSE r{r}] {'pre-collapse' if r < collapse_round else 'POST-COLLAPSE'}",
                make_embedding(a0_center, noise, emb_dim),
                make_hidden_state(a0_hs, noise * 0.5, hs_dim),
                ts,
                random.randint(2000, 4000)
            ),
            make_turn(
                "agent_1",
                f"[SUDDEN_COLLAPSE r{r}] {'pre-collapse' if r < collapse_round else 'POST-COLLAPSE'}",
                make_embedding(a1_center, noise, emb_dim),
                make_hidden_state(a1_hs, noise * 0.5, hs_dim),
                ts + random.randint(2000, 5000),
                random.randint(2000, 4000)
            ),
        ]

        rounds.append({"round": r, "turns": turns})

    return {
        "session_id": "synthetic_sudden_collapse_01",
        "task_type": "synthetic_validation",
        "task_prompt": "Synthetic sudden collapse for H2 falsification testing",
        "communication_condition": "natural_sequential",
        "agent_configs": [
            {"agent_id": "agent_0", "system_prompt": "collapse_A", "temperature": 0.7, "top_k": 50},
            {"agent_id": "agent_1", "system_prompt": "collapse_B", "temperature": 0.7, "top_k": 50},
        ],
        "rounds": rounds,
        "metadata": {
            "model": "synthetic",
            "num_rounds": n_rounds,
            "collapse_round": collapse_round,
            "scenario": "sudden_collapse",
            "expected_pid": "sudden redundancy spike at collapse, no gradual buildup",
            "expected_entropy": "stable H(t) then cliff — NO preceding δ(t) decline",
            "h2_prediction": "FALSIFIED (cascade dynamics, not phase transition)",
            "date": "2026-03-02"
        }
    }


# ============================================================
# SCENARIO 4: GRADUAL CONVERGENCE (H2 confirmation test)
# ============================================================

def generate_gradual_convergence(n_rounds: int = 10, seed: int = 789) -> dict:
    """
    Agents converge slowly with δ(t) declining BEFORE H(t) collapses.
    Simulates critical slowing down: agents start repeating themselves
    (small δ) before the group entropy actually drops.
    
    Implementation: agent positions move toward merge point, but the PER-TURN
    variation (δ) decreases faster than the BETWEEN-AGENT distance.
    
    Expected PID: gradual redundancy rise, gradual synergy decline.
    Expected H(t): delayed monotonic decrease.
    Expected δ(t): early decline (2+ rounds before H(t) decline).
    """
    np.random.seed(seed)
    random.seed(seed)

    emb_dim = 384
    hs_dim = 2048

    center_A = np.random.normal(0, 1, emb_dim)
    center_A = center_A / np.linalg.norm(center_A)
    center_B = np.random.normal(0, 1, emb_dim)
    center_B = center_B / np.linalg.norm(center_B)
    center_merged = (center_A + center_B) / 2
    center_merged = center_merged / np.linalg.norm(center_merged)

    hs_A = np.random.normal(0, 1, hs_dim)
    hs_B = np.random.normal(0, 1, hs_dim)
    hs_merged = (hs_A + hs_B) / 2

    rounds = []
    base_ts = 1709341200000

    for r in range(n_rounds):
        # Position convergence: SLOW (agents stay apart until later rounds)
        # Using sigmoid for delayed convergence
        pos_alpha = 1 / (1 + np.exp(-(r - n_rounds * 0.7) * 2))

        # δ convergence: FAST (agents start repeating themselves early)
        # Noise/variation decreases much earlier than position converges
        delta_decay = np.exp(-r * 0.4)  # Exponential decay starting from round 0

        a0_center = (1 - pos_alpha) * center_A + pos_alpha * center_merged
        a1_center = (1 - pos_alpha) * center_B + pos_alpha * center_merged

        a0_hs = (1 - pos_alpha) * hs_A + pos_alpha * hs_merged
        a1_hs = (1 - pos_alpha) * hs_B + pos_alpha * hs_merged

        # Key: noise (which determines δ between consecutive turns) decays FAST
        noise = 0.1 * delta_decay

        ts = base_ts + r * 30000
        turns = [
            make_turn(
                "agent_0",
                f"[GRADUAL r{r}] pos_alpha={pos_alpha:.3f} delta_decay={delta_decay:.3f}",
                make_embedding(a0_center, noise, emb_dim),
                make_hidden_state(a0_hs, noise * 0.5, hs_dim),
                ts,
                random.randint(2000, 4000)
            ),
            make_turn(
                "agent_1",
                f"[GRADUAL r{r}] pos_alpha={pos_alpha:.3f} delta_decay={delta_decay:.3f}",
                make_embedding(a1_center, noise, emb_dim),
                make_hidden_state(a1_hs, noise * 0.5, hs_dim),
                ts + random.randint(2000, 5000),
                random.randint(2000, 4000)
            ),
        ]

        rounds.append({"round": r, "turns": turns})

    return {
        "session_id": "synthetic_gradual_convergence_01",
        "task_type": "synthetic_validation",
        "task_prompt": "Synthetic gradual convergence for H2 confirmation testing",
        "communication_condition": "natural_sequential",
        "agent_configs": [
            {"agent_id": "agent_0", "system_prompt": "gradual_A", "temperature": 0.7, "top_k": 50},
            {"agent_id": "agent_1", "system_prompt": "gradual_B", "temperature": 0.7, "top_k": 50},
        ],
        "rounds": rounds,
        "metadata": {
            "model": "synthetic",
            "num_rounds": n_rounds,
            "scenario": "gradual_convergence",
            "expected_pid": "gradual redundancy rise, gradual synergy decline",
            "expected_entropy": "δ(t) declines 2+ rounds before H(t) collapse",
            "h2_prediction": "CONFIRMED (critical slowing down present)",
            "date": "2026-03-02"
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic debate traces for pipeline validation")
    parser.add_argument("--output_dir", default="./synthetic_traces", help="Output directory")
    parser.add_argument("--n_rounds", type=int, default=10, help="Number of rounds per debate")
    parser.add_argument("--scenario", choices=["converging", "diverging", "sudden_collapse", "gradual", "all"],
                        default="all", help="Which scenario to generate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generators = {
        "converging": lambda: generate_converging(args.n_rounds, args.seed),
        "diverging": lambda: generate_diverging(args.n_rounds, args.seed + 100),
        "sudden_collapse": lambda: generate_sudden_collapse(args.n_rounds, seed=args.seed + 200),
        "gradual": lambda: generate_gradual_convergence(args.n_rounds, args.seed + 300),
    }

    scenarios = list(generators.keys()) if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        trace = generators[scenario]()
        out_path = os.path.join(args.output_dir, f"{trace['session_id']}.json")
        with open(out_path, 'w') as f:
            json.dump(trace, f, indent=2)
        print(f"✅ Generated: {out_path}")
        print(f"   Scenario: {scenario}")
        print(f"   Expected PID: {trace['metadata']['expected_pid']}")
        print(f"   Expected entropy: {trace['metadata']['expected_entropy']}")
        print()

    print(f"\n🔬 Generated {len(scenarios)} synthetic traces in {args.output_dir}")
    print("\nValidation checklist:")
    print("  1. Run pid_analysis.py on these traces")
    print("  2. Run measure_entropy.py on these traces")
    print("  3. Check: converging → rising redundancy, falling synergy? ✓/✗")
    print("  4. Check: diverging → stable synergy, low redundancy? ✓/✗")
    print("  5. Check: sudden_collapse → H(t) cliff WITHOUT preceding δ(t) decline? ✓/✗")
    print("  6. Check: gradual → δ(t) decline 2+ rounds BEFORE H(t) collapse? ✓/✗")
    print("\nIf all 4 pass → pipeline validated. Proceed to real OLMo sessions.")


if __name__ == "__main__":
    main()
