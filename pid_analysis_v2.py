#!/usr/bin/env python3
"""
pid_analysis_v2.py — PID Analysis with Binary Discretization
Paper A, Hypothesis H4: PID–Entropy Convergent Validity

v2 FIX: Binary discretization solves the degenerate distribution problem from v1.

The Problem (v1):
    With 10 rounds × 2 agents and k=3 clusters, we get 9 transition samples across
    up to 27 possible (X, Y, Z) triples. Almost everything observed at most once →
    flat empirical distribution → PID returns zeros.

The Fix (v2):
    Each agent at each round gets a single bit:
        1 = "moved toward" the other agent (cosine similarity increased)
        0 = "moved away" (cosine similarity decreased or stayed)

    This gives us:
        - 2 possible X values × 2 possible Y values × 4 possible Z values = 16 triples
        - With 9 transitions from a 10-round session: meaningful repeated observations
        - Directly interpretable: convergent vs divergent moves map onto our research question

    For pooled analysis (--pool), we aggregate transitions across sessions within the
    same condition type, giving even richer distributions.

Requirements:
    pip install dit numpy sentence-transformers scipy matplotlib

Usage:
    # Per-session analysis
    python pid_analysis_v2.py --traces_dir ./debate_traces --output_dir ./pid_results --plot

    # Pooled by condition (recommended for richer distributions)
    python pid_analysis_v2.py --traces_dir ./debate_traces --output_dir ./pid_results --pool --plot

    # H4 test
    python pid_analysis_v2.py --traces_dir ./debate_traces --entropy_file ./entropy_results.json --test_h4 --plot

Author: Herald (Paper A collaboration with Chaewon)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

try:
    import dit
    from dit.pid import PID_WB
    HAS_DIT = True
except ImportError:
    HAS_DIT = False
    print("WARNING: `dit` not found. Install: pip install dit")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_debate_traces(traces_dir: str) -> list[dict]:
    """Load debate traces from JSON files."""
    traces = []
    traces_path = Path(traces_dir)
    for f in sorted(traces_path.glob("*.json")):
        with open(f) as fh:
            traces.append(json.load(fh))
    print(f"Loaded {len(traces)} debate traces from {traces_dir}")
    return traces


def compute_embeddings_if_needed(traces: list[dict], model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """Embed turn texts if embeddings not already present."""
    sample_turn = traces[0]["rounds"][0]["turns"][0]
    if "embedding" in sample_turn and sample_turn["embedding"] is not None:
        print("Embeddings already present in traces.")
        return traces
    if not HAS_ST:
        raise RuntimeError("No embeddings in traces and sentence-transformers not installed.")
    print(f"Computing embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    for trace in traces:
        for round_data in trace["rounds"]:
            texts = [turn["text"] for turn in round_data["turns"]]
            embeddings = model.encode(texts)
            for turn, emb in zip(round_data["turns"], embeddings):
                turn["embedding"] = emb.tolist()
    return traces


# ============================================================
# 2. BINARY DISCRETIZATION
# ============================================================

def compute_binary_moves(trace: dict) -> dict[str, list[int]]:
    """
    For each agent at each round t (t >= 1), compute a binary move:
        1 = "convergent" — agent moved TOWARD the other agent's previous position
        0 = "divergent"  — agent moved AWAY or stayed

    Convergence is measured as:
        cos_sim(agent_i[t], agent_j[t-1]) > cos_sim(agent_i[t-1], agent_j[t-1])

    i.e., agent i's new response is more similar to agent j's last response
    than agent i's own previous response was.

    Returns:
        dict mapping agent_id → list of binary moves (length = n_rounds - 1)
    """
    # Extract per-agent embedding sequences
    agent_embeddings = {}
    for round_data in trace["rounds"]:
        for turn in round_data["turns"]:
            aid = turn["agent_id"]
            if aid not in agent_embeddings:
                agent_embeddings[aid] = []
            agent_embeddings[aid].append(np.array(turn["embedding"]))

    agents = sorted(agent_embeddings.keys())
    if len(agents) < 2:
        return {a: [] for a in agents}

    n_rounds = min(len(agent_embeddings[a]) for a in agents)
    moves = {a: [] for a in agents}

    for t in range(1, n_rounds):
        for i, agent_i in enumerate(agents):
            # The "other" agent — for 2 agents this is simple
            agent_j = agents[1 - i] if len(agents) == 2 else agents[(i + 1) % len(agents)]

            emb_i_now = agent_embeddings[agent_i][t]
            emb_i_prev = agent_embeddings[agent_i][t - 1]
            emb_j_prev = agent_embeddings[agent_j][t - 1]

            # Cosine similarity (1 - cosine distance)
            sim_now = 1.0 - cosine_dist(emb_i_now, emb_j_prev)
            sim_prev = 1.0 - cosine_dist(emb_i_prev, emb_j_prev)

            # Binary: did agent_i move toward agent_j?
            move = 1 if sim_now > sim_prev else 0
            moves[agent_i].append(move)

    return moves


# ============================================================
# 3. PID COMPUTATION (Binary Version)
# ============================================================

def compute_pid_binary(transitions: list[tuple[str, str, str]]) -> dict:
    """
    Compute PID from a list of (X, Y, Z) transition triples.

    X = agent_0 binary move at round t
    Y = agent_1 binary move at round t
    Z = combined next-round state (agent_0_move[t+1], agent_1_move[t+1])

    With binary variables:
        X in {0, 1}, Y in {0, 1}, Z in {00, 01, 10, 11}
        Total possible triples: 2 x 2 x 4 = 16

    This is tractable even with 8-9 samples from a 10-round session.
    """
    if not HAS_DIT:
        return {"synergy": np.nan, "redundancy": np.nan,
                "unique_0": np.nan, "unique_1": np.nan, "error": "dit not installed"}

    if len(transitions) < 2:
        return {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0,
                "note": "insufficient_transitions"}

    # Build empirical distribution
    counts = {}
    for triple in transitions:
        counts[triple] = counts.get(triple, 0) + 1

    total = sum(counts.values())
    outcomes = list(counts.keys())
    probs = [counts[k] / total for k in outcomes]

    # Need at least 2 distinct outcomes for meaningful PID
    if len(outcomes) < 2:
        return {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0,
                "note": "degenerate_single_outcome"}

    try:
        d = dit.Distribution(outcomes, probs)
        d.set_rv_names(['X', 'Y', 'Z'])
        pid = PID_WB(d, ['X', 'Y'], 'Z')

        return {
            "synergy": float(pid.get_partial(((0, 1),))),
            "redundancy": float(pid.get_partial(((0,), (1,)))),
            "unique_0": float(pid.get_partial(((0,),))),
            "unique_1": float(pid.get_partial(((1,),))),
            "n_transitions": len(transitions),
            "n_unique_triples": len(outcomes),
        }
    except Exception as e:
        return {"synergy": np.nan, "redundancy": np.nan,
                "unique_0": np.nan, "unique_1": np.nan, "error": str(e)}


def compute_pid_trajectory_binary(trace: dict) -> list[dict]:
    """
    Compute PID at each round using cumulative binary transitions.

    At round t, we use all transitions from [1..t] to build the distribution.
    This gives increasingly stable PID estimates as the debate progresses.
    """
    moves = compute_binary_moves(trace)
    agents = sorted(moves.keys())

    if len(agents) < 2:
        return []

    a0_moves = moves[agents[0]]
    a1_moves = moves[agents[1]]
    n = min(len(a0_moves), len(a1_moves))

    trajectory = []

    for t in range(n):
        # Build transitions up to round t
        # Each transition: (X=a0[t'], Y=a1[t'], Z=combined_next[t'+1])
        transitions = []
        for t_prime in range(min(t, n - 1)):  # need t'+1 to exist
            x = str(a0_moves[t_prime])
            y = str(a1_moves[t_prime])
            z = f"{a0_moves[t_prime + 1]}{a1_moves[t_prime + 1]}"
            transitions.append((x, y, z))

        if len(transitions) >= 2:
            pid = compute_pid_binary(transitions)
        else:
            pid = {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0,
                   "note": "warming_up"}

        pid["round"] = t + 1  # +1 because moves start at round 1
        pid["session_id"] = trace.get("session_id", "unknown")
        pid["cumulative_transitions"] = len(transitions)

        # Also record the raw binary moves for this round
        pid["agent_0_move"] = a0_moves[t]
        pid["agent_1_move"] = a1_moves[t]
        pid["convergent_fraction"] = (
            sum(a0_moves[:t+1] + a1_moves[:t+1]) / (2 * (t + 1))
        )

        trajectory.append(pid)

    return trajectory


def compute_pid_pooled(traces: list[dict]) -> dict:
    """
    Pool transitions across sessions within the same condition.

    This gives much richer distributions (e.g., 30 sessions x 9 transitions = 270 samples)
    and is the recommended mode for the full experiment.

    Returns dict mapping condition -> PID results.
    """
    condition_transitions = defaultdict(list)

    for trace in traces:
        condition = trace.get("communication_condition", "unknown")
        moves = compute_binary_moves(trace)
        agents = sorted(moves.keys())

        if len(agents) < 2:
            continue

        a0_moves = moves[agents[0]]
        a1_moves = moves[agents[1]]
        n = min(len(a0_moves), len(a1_moves))

        for t in range(n - 1):
            x = str(a0_moves[t])
            y = str(a1_moves[t])
            z = f"{a0_moves[t + 1]}{a1_moves[t + 1]}"
            condition_transitions[condition].append((x, y, z))

    results = {}
    for condition, transitions in condition_transitions.items():
        print(f"  Condition '{condition}': {len(transitions)} pooled transitions")
        results[condition] = compute_pid_binary(transitions)
        results[condition]["condition"] = condition

    return results


# ============================================================
# 4. H4 TESTING — PID-Entropy Convergent Validity
# ============================================================

def load_entropy_results(entropy_file: str) -> dict:
    """Load entropy H(t) results from measure_entropy.py output."""
    with open(entropy_file) as f:
        return json.load(f)


def test_h4(pid_results: dict, entropy_results: dict) -> dict:
    """
    Test H4: PID redundancy increase correlates with entropy decrease.

    Per session:
    - Redundancy slope (from PID trajectory) -- expected positive (increasing)
    - H(t) slope (from entropy trajectory) -- expected negative (decreasing)
    - Correlation across sessions: expected r < -0.5

    Also computes convergent_fraction correlation with entropy as a sanity check.
    """
    redundancy_slopes = []
    entropy_slopes = []
    synergy_slopes = []
    convergence_fracs = []
    session_ids = []

    for sid, pid_traj in pid_results.items():
        if sid not in entropy_results:
            continue

        # Filter rounds with valid PID
        valid = [p for p in pid_traj if not np.isnan(p.get("redundancy", np.nan))]
        if len(valid) < 3:
            continue

        rounds = [p["round"] for p in valid]
        redundancies = [p["redundancy"] for p in valid]
        synergies = [p["synergy"] for p in valid]

        r_slope = stats.linregress(rounds, redundancies).slope
        s_slope = stats.linregress(rounds, synergies).slope

        # Final convergent fraction
        conv_frac = valid[-1].get("convergent_fraction", 0.5)

        # Entropy trajectory
        H_traj = entropy_results[sid].get("H_trajectory", [])
        if len(H_traj) < 3:
            continue
        h_slope = stats.linregress(range(len(H_traj)), H_traj).slope

        redundancy_slopes.append(r_slope)
        synergy_slopes.append(s_slope)
        entropy_slopes.append(h_slope)
        convergence_fracs.append(conv_frac)
        session_ids.append(sid)

    if len(redundancy_slopes) < 5:
        return {
            "status": "insufficient_data",
            "n_sessions": len(redundancy_slopes),
            "message": "Need at least 5 sessions with both PID and entropy data"
        }

    # Core H4 test
    r_corr, r_p = stats.pearsonr(redundancy_slopes, entropy_slopes)
    s_corr, s_p = stats.spearmanr(redundancy_slopes, entropy_slopes)
    syn_corr, syn_p = stats.pearsonr(synergy_slopes, entropy_slopes)

    # Sanity check: convergent fraction vs entropy slope
    cf_corr, cf_p = stats.pearsonr(convergence_fracs, entropy_slopes)

    return {
        "status": "complete",
        "n_sessions": len(redundancy_slopes),
        "h4_redundancy_entropy_pearson_r": round(r_corr, 4),
        "h4_redundancy_entropy_pearson_p": round(r_p, 6),
        "h4_redundancy_entropy_spearman_r": round(s_corr, 4),
        "h4_redundancy_entropy_spearman_p": round(s_p, 6),
        "h4_supported": r_corr < -0.5 and r_p < 0.05,
        "synergy_entropy_pearson_r": round(syn_corr, 4),
        "synergy_entropy_pearson_p": round(syn_p, 6),
        "convergent_frac_entropy_r": round(cf_corr, 4),
        "convergent_frac_entropy_p": round(cf_p, 6),
        "interpretation": (
            "H4 SUPPORTED -- redundancy increase correlates with entropy decrease"
            if (r_corr < -0.5 and r_p < 0.05)
            else "H4 NOT SUPPORTED -- no significant negative correlation found"
        ),
        "detail": {
            "redundancy_slopes": [round(x, 6) for x in redundancy_slopes],
            "synergy_slopes": [round(x, 6) for x in synergy_slopes],
            "entropy_slopes": [round(x, 6) for x in entropy_slopes],
            "convergence_fracs": [round(x, 4) for x in convergence_fracs],
            "session_ids": session_ids,
        }
    }


# ============================================================
# 5. VISUALIZATION
# ============================================================

def plot_pid_trajectory(pid_traj: list[dict], session_id: str, output_dir: str):
    """Plot PID components + convergent fraction over rounds."""
    if not HAS_PLT:
        return

    rounds = [p["round"] for p in pid_traj]
    synergy = [p.get("synergy", 0) for p in pid_traj]
    redundancy = [p.get("redundancy", 0) for p in pid_traj]
    unique_0 = [p.get("unique_0", 0) for p in pid_traj]
    unique_1 = [p.get("unique_1", 0) for p in pid_traj]
    conv_frac = [p.get("convergent_fraction", 0.5) for p in pid_traj]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(rounds, synergy, 'b-o', label='Synergy', linewidth=2, markersize=5)
    ax1.plot(rounds, redundancy, 'r-s', label='Redundancy', linewidth=2, markersize=5)
    ax1.plot(rounds, unique_0, 'g--^', label='Unique (Agent 0)', alpha=0.7, markersize=4)
    ax1.plot(rounds, unique_1, 'm--v', label='Unique (Agent 1)', alpha=0.7, markersize=4)
    ax1.set_ylabel('Information (bits)', fontsize=12)
    ax1.set_title(f'PID Trajectory (Binary) -- {session_id}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(rounds, conv_frac, 0.5, alpha=0.3,
                     color='green', where=[c >= 0.5 for c in conv_frac])
    ax2.fill_between(rounds, conv_frac, 0.5, alpha=0.3,
                     color='red', where=[c < 0.5 for c in conv_frac])
    ax2.plot(rounds, conv_frac, 'k-o', linewidth=1.5, markersize=4)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Convergent Fraction', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"pid_binary_{session_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pooled_pid(pooled_results: dict, output_dir: str):
    """Bar chart comparing PID components across conditions."""
    if not HAS_PLT:
        return

    conditions = sorted(pooled_results.keys())
    synergy = [pooled_results[c].get("synergy", 0) for c in conditions]
    redundancy = [pooled_results[c].get("redundancy", 0) for c in conditions]
    unique_0 = [pooled_results[c].get("unique_0", 0) for c in conditions]
    unique_1 = [pooled_results[c].get("unique_1", 0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, synergy, width, label='Synergy', color='steelblue')
    ax.bar(x - 0.5*width, redundancy, width, label='Redundancy', color='indianred')
    ax.bar(x + 0.5*width, unique_0, width, label='Unique (A0)', color='seagreen', alpha=0.8)
    ax.bar(x + 1.5*width, unique_1, width, label='Unique (A1)', color='mediumpurple', alpha=0.8)

    ax.set_xlabel('Communication Condition', fontsize=12)
    ax.set_ylabel('Information (bits)', fontsize=12)
    ax.set_title('Pooled PID by Condition', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, "pid_pooled_conditions.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_h4_correlation(h4_results: dict, output_dir: str):
    """Scatter: redundancy slope vs entropy slope."""
    if not HAS_PLT or h4_results["status"] != "complete":
        return

    detail = h4_results["detail"]
    r_slopes = detail["redundancy_slopes"]
    h_slopes = detail["entropy_slopes"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_slopes, h_slopes, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)

    m, b = np.polyfit(r_slopes, h_slopes, 1)
    x_line = np.linspace(min(r_slopes), max(r_slopes), 100)
    ax.plot(x_line, m * x_line + b, 'r--', linewidth=2, alpha=0.7)

    r_val = h4_results["h4_redundancy_entropy_pearson_r"]
    p_val = h4_results["h4_redundancy_entropy_pearson_p"]

    ax.set_xlabel('Redundancy Slope (per round)', fontsize=12)
    ax.set_ylabel('Entropy H(t) Slope (per round)', fontsize=12)
    ax.set_title(f'H4: PID Redundancy vs Entropy\nr={r_val:.3f}, p={p_val:.4f}', fontsize=14)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "h4_correlation.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PID Analysis v2 -- Binary Discretization for Multi-Agent Debates")
    parser.add_argument("--traces_dir", required=True,
                        help="Directory containing debate trace JSON files")
    parser.add_argument("--output_dir", default="./pid_results",
                        help="Output directory")
    parser.add_argument("--entropy_file", default=None,
                        help="Path to entropy results JSON (for H4 test)")
    parser.add_argument("--test_h4", action="store_true",
                        help="Run H4 convergent validity test")
    parser.add_argument("--pool", action="store_true",
                        help="Pool transitions across sessions by condition")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2",
                        help="Sentence embedding model")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not HAS_DIT:
        print("\nERROR: `dit` required. pip install dit")
        sys.exit(1)

    # Step 1: Load
    print("\n=== Step 1: Loading debate traces ===")
    traces = load_debate_traces(args.traces_dir)

    # Step 2: Embeddings
    print("\n=== Step 2: Checking embeddings ===")
    traces = compute_embeddings_if_needed(traces, args.embedding_model)

    # Step 3: Binary discretization + PID
    print("\n=== Step 3: Computing binary moves + PID ===")

    if args.pool:
        print("  Mode: pooled by condition")
        pooled = compute_pid_pooled(traces)

        out_file = os.path.join(args.output_dir, "pid_pooled.json")
        with open(out_file, 'w') as f:
            json.dump(pooled, f, indent=2, default=str)
        print(f"\nPooled PID saved to {out_file}")

        if args.plot:
            plot_pooled_pid(pooled, args.output_dir)

    # Per-session trajectories (always computed for H4)
    print("  Mode: per-session trajectories")
    pid_results = {}
    for trace in traces:
        sid = trace.get("session_id", "unknown")
        print(f"    {sid}...")
        traj = compute_pid_trajectory_binary(trace)
        pid_results[sid] = traj

    # Serialize
    serializable = {}
    for sid, traj in pid_results.items():
        serializable[sid] = []
        for entry in traj:
            clean = {}
            for k, v in entry.items():
                if isinstance(v, (np.floating, float)):
                    clean[k] = None if np.isnan(v) else round(float(v), 6)
                elif isinstance(v, np.integer):
                    clean[k] = int(v)
                else:
                    clean[k] = v
            serializable[sid].append(clean)

    traj_file = os.path.join(args.output_dir, "pid_trajectories_v2.json")
    with open(traj_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nPID trajectories saved to {traj_file}")

    # Step 4: Plots
    if args.plot:
        print("\n=== Step 4: Generating plots ===")
        for sid, traj in pid_results.items():
            if traj:
                plot_pid_trajectory(traj, sid, args.output_dir)

    # Step 5: H4 test
    if args.test_h4:
        if not args.entropy_file:
            print("\nWARNING: --test_h4 requires --entropy_file. Skipping.")
        else:
            print("\n=== Step 5: Testing H4 (PID-Entropy Convergent Validity) ===")
            entropy_results = load_entropy_results(args.entropy_file)
            h4 = test_h4(pid_results, entropy_results)

            h4_file = os.path.join(args.output_dir, "h4_results_v2.json")
            with open(h4_file, 'w') as f:
                json.dump(h4, f, indent=2, default=str)

            print(f"\n{'='*60}")
            print(f"  H4 RESULT: {h4['interpretation']}")
            if h4['status'] == 'complete':
                print(f"  Pearson r = {h4['h4_redundancy_entropy_pearson_r']}")
                print(f"  p-value   = {h4['h4_redundancy_entropy_pearson_p']}")
                print(f"  N sessions = {h4['n_sessions']}")
            print(f"{'='*60}")

            if args.plot:
                plot_h4_correlation(h4, args.output_dir)

    print("\n=== PID v2 analysis complete. ===")


if __name__ == "__main__":
    main()
