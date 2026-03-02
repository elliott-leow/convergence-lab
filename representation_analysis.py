#!/usr/bin/env python3
"""
representation_analysis.py — Hidden State Trajectory Analysis
Mechanistic analysis of divergence using OLMo 2's internal representations.

Only possible with open-weight models — this is the analysis closed-model
papers can't do.

Metrics computed:
1. Inter-agent representation distance: cosine distance between agents'
   hidden states at each round. Divergence = increasing distance.
2. Intra-agent drift: how much each agent's representation changes from
   their own previous round. High drift = large internal state changes.
3. Convergence ratio: inter-agent distance / mean intra-agent drift.
   >1 = representations diverging faster than individual agents are changing.
   <1 = representations converging despite individual change.
4. Cross-agent predictability: cosine similarity between agent A's delta
   and agent B's delta (same as v3 coupling, but in representation space).

Expected trace format:
{
    "rounds": [
        {
            "round": 0,
            "turns": [
                {
                    "agent_id": "agent_a",
                    "text": "...",
                    "embedding": [...],           # output embedding (e.g., MiniLM)
                    "hidden_state": [...]          # OLMo final-layer mean, 4096-dim
                },
                ...
            ]
        }
    ]
}

Usage:
    python representation_analysis.py --traces_dir ./traces --output_dir ./repr_results --plot

Author: Herald (Paper A collaboration with Chaewon)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

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

def load_traces(traces_dir: str) -> list[dict]:
    """Load debate traces from JSON files."""
    traces = []
    for f in sorted(Path(traces_dir).glob("*.json")):
        with open(f) as fh:
            trace = json.load(fh)
            # Verify hidden states exist
            sample = trace["rounds"][0]["turns"][0]
            # Support both formats: "hidden_state" or nested "hidden_states.final_layer_mean"
            hs = sample.get("hidden_state") or (sample.get("hidden_states", {}) or {}).get("final_layer_mean")
            if hs is None:
                print(f"  SKIP {f.name}: no hidden_state data")
                continue
            traces.append(trace)
    print(f"Loaded {len(traces)} traces with hidden states from {traces_dir}")
    return traces


def extract_agent_data(trace: dict) -> dict:
    """
    Extract per-agent hidden states and embeddings.
    Returns dict: {agent_id: {"hidden": [np.array...], "embedding": [np.array...]}}
    """
    agents = {}
    for rd in trace["rounds"]:
        for turn in rd["turns"]:
            aid = turn["agent_id"]
            if aid not in agents:
                agents[aid] = {"hidden": [], "embedding": []}
            hs = turn.get("hidden_state") or (turn.get("hidden_states", {}) or {}).get("final_layer_mean")
            agents[aid]["hidden"].append(np.array(hs))
            if turn.get("embedding"):
                agents[aid]["embedding"].append(np.array(turn["embedding"]))
    return agents


# ============================================================
# 2. REPRESENTATION METRICS
# ============================================================

def safe_cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance with zero-vector handling."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0  # maximally distant
    return float(cosine_dist(a, b))


def safe_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - safe_cosine_dist(a, b)


def compute_representation_trajectory(trace: dict) -> dict:
    """
    Compute all representation-level metrics for one debate session.

    Returns dict with per-round metrics + summary statistics.
    """
    agents = extract_agent_data(trace)
    agent_ids = sorted(agents.keys())

    if len(agent_ids) < 2:
        return {"error": "need at least 2 agents"}

    a0 = agents[agent_ids[0]]
    a1 = agents[agent_ids[1]]
    n_rounds = min(len(a0["hidden"]), len(a1["hidden"]))

    rounds = []

    for t in range(n_rounds):
        h0 = a0["hidden"][t]
        h1 = a1["hidden"][t]

        # 1. Inter-agent hidden state distance
        inter_dist = safe_cosine_dist(h0, h1)

        # 2. Intra-agent drift (from previous round)
        drift_0 = safe_cosine_dist(h0, a0["hidden"][t-1]) if t > 0 else 0.0
        drift_1 = safe_cosine_dist(h1, a1["hidden"][t-1]) if t > 0 else 0.0
        mean_drift = (drift_0 + drift_1) / 2

        # 3. Convergence ratio
        conv_ratio = inter_dist / mean_drift if mean_drift > 1e-10 else float('inf')

        # 4. Representation coupling (delta similarity)
        if t > 0:
            delta_0 = h0 - a0["hidden"][t-1]
            delta_1 = h1 - a1["hidden"][t-1]
            repr_coupling = safe_cosine_sim(delta_0, delta_1)
        else:
            repr_coupling = 0.0

        # 5. If we have output embeddings too, compare representation vs output divergence
        emb_dist = None
        if a0["embedding"] and a1["embedding"] and t < len(a0["embedding"]):
            emb_dist = safe_cosine_dist(a0["embedding"][t], a1["embedding"][t])

        # 6. Hidden state norms (are representations growing/shrinking?)
        norm_0 = float(np.linalg.norm(h0))
        norm_1 = float(np.linalg.norm(h1))

        round_data = {
            "round": t,
            "inter_agent_distance": round(inter_dist, 6),
            "intra_agent_drift_0": round(drift_0, 6),
            "intra_agent_drift_1": round(drift_1, 6),
            "mean_intra_drift": round(mean_drift, 6),
            "convergence_ratio": round(conv_ratio, 6) if conv_ratio != float('inf') else None,
            "repr_coupling": round(repr_coupling, 6),
            "hidden_norm_0": round(norm_0, 4),
            "hidden_norm_1": round(norm_1, 4),
        }
        if emb_dist is not None:
            round_data["embedding_distance"] = round(emb_dist, 6)

        rounds.append(round_data)

    # Summary statistics
    inter_dists = [r["inter_agent_distance"] for r in rounds]
    couplings = [r["repr_coupling"] for r in rounds[1:]]  # skip round 0

    # Monotonicity of inter-agent distance
    if len(inter_dists) >= 3:
        rho, rho_p = stats.spearmanr(range(len(inter_dists)), inter_dists)
    else:
        rho, rho_p = 0, 1

    # Compare representation divergence to embedding divergence
    repr_vs_emb = None
    if rounds[0].get("embedding_distance") is not None:
        emb_dists = [r["embedding_distance"] for r in rounds if r.get("embedding_distance") is not None]
        if len(emb_dists) >= 3 and len(inter_dists) >= 3:
            min_len = min(len(emb_dists), len(inter_dists))
            corr, corr_p = stats.pearsonr(inter_dists[:min_len], emb_dists[:min_len])
            repr_vs_emb = {
                "pearson_r": round(corr, 4),
                "pearson_p": round(corr_p, 6),
                "interpretation": (
                    "Representation and output divergence are correlated — deep mechanism"
                    if corr > 0.5 and corr_p < 0.1 else
                    "Representation and output divergence are decoupled — surface phenomenon"
                    if corr < 0.3 else
                    "Moderate correlation"
                ),
            }

    summary = {
        "session_id": trace.get("session_id", "unknown"),
        "n_rounds": n_rounds,
        "inter_distance_rho": round(rho, 4),
        "inter_distance_rho_p": round(rho_p, 6),
        "inter_distance_trend": (
            "diverging" if rho > 0.5 and rho_p < 0.1 else
            "converging" if rho < -0.5 and rho_p < 0.1 else
            "flat"
        ),
        "mean_coupling": round(float(np.mean(couplings)), 4) if couplings else None,
        "initial_inter_distance": inter_dists[0] if inter_dists else None,
        "final_inter_distance": inter_dists[-1] if inter_dists else None,
        "repr_vs_embedding": repr_vs_emb,
    }

    return {"rounds": rounds, "summary": summary}


# ============================================================
# 3. CROSS-SESSION ANALYSIS
# ============================================================

def analyze_all(traces: list[dict]) -> dict:
    """Run representation analysis across all traces."""
    results = {}
    summaries = []

    for trace in traces:
        sid = trace.get("session_id", "unknown")
        print(f"  Analyzing {sid}...")
        result = compute_representation_trajectory(trace)
        results[sid] = result
        if "summary" in result:
            summaries.append(result["summary"])

    # Cross-session summary
    if summaries:
        rhos = [s["inter_distance_rho"] for s in summaries]
        trends = [s["inter_distance_trend"] for s in summaries]

        cross_session = {
            "n_sessions": len(summaries),
            "mean_rho": round(float(np.mean(rhos)), 4),
            "diverging_count": trends.count("diverging"),
            "converging_count": trends.count("converging"),
            "flat_count": trends.count("flat"),
            "repr_diverges_like_output": all(
                s.get("repr_vs_embedding", {}).get("pearson_r", 0) > 0.5
                for s in summaries if s.get("repr_vs_embedding")
            ),
        }
    else:
        cross_session = {"n_sessions": 0}

    return {"sessions": results, "cross_session": cross_session}


# ============================================================
# 4. VISUALIZATION
# ============================================================

def plot_representation_trajectory(result: dict, session_id: str, output_dir: str):
    """Plot inter-agent distance + coupling over rounds."""
    if not HAS_PLT:
        return

    rounds_data = result["rounds"]
    rounds = [r["round"] for r in rounds_data]
    inter_dist = [r["inter_agent_distance"] for r in rounds_data]
    coupling = [r["repr_coupling"] for r in rounds_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Inter-agent distance
    ax1.plot(rounds, inter_dist, 'r-o', linewidth=2, markersize=5, label='Hidden state distance')
    if rounds_data[0].get("embedding_distance") is not None:
        emb_dist = [r.get("embedding_distance", 0) for r in rounds_data]
        ax1.plot(rounds, emb_dist, 'b--s', linewidth=2, markersize=5, alpha=0.7, label='Embedding distance')
    ax1.set_ylabel('Cosine Distance', fontsize=12)
    ax1.set_title(f'Representation Trajectories — {session_id}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Representation coupling
    ax2.plot(rounds, coupling, 'g-^', linewidth=2, markersize=5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(rounds, coupling, 0, alpha=0.2,
                     color='green', where=[c >= 0 for c in coupling])
    ax2.fill_between(rounds, coupling, 0, alpha=0.2,
                     color='red', where=[c < 0 for c in coupling])
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Representation Coupling', fontsize=12)
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"repr_{session_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_summary(cross_session: dict, all_results: dict, output_dir: str):
    """Summary plot: representation vs embedding divergence across sessions."""
    if not HAS_PLT:
        return

    sessions = all_results["sessions"]
    repr_rhos = []
    emb_rhos = []
    labels = []

    for sid, result in sessions.items():
        if "summary" not in result:
            continue
        summary = result["summary"]
        repr_rhos.append(summary["inter_distance_rho"])
        labels.append(sid[:20])

        # If we have embedding distance trajectory, compute its rho too
        rounds_data = result["rounds"]
        emb_dists = [r.get("embedding_distance") for r in rounds_data
                     if r.get("embedding_distance") is not None]
        if len(emb_dists) >= 3:
            emb_rho, _ = stats.spearmanr(range(len(emb_dists)), emb_dists)
            emb_rhos.append(emb_rho)
        else:
            emb_rhos.append(None)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(repr_rhos))

    ax.bar([i - 0.15 for i in x], repr_rhos, 0.3, label='Hidden State ρ', color='indianred')
    valid_emb = [(i, r) for i, r in zip(x, emb_rhos) if r is not None]
    if valid_emb:
        ax.bar([i + 0.15 for i, _ in valid_emb], [r for _, r in valid_emb], 0.3,
               label='Embedding ρ', color='steelblue')

    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Spearman ρ (distance over rounds)', fontsize=12)
    ax.set_title('Representation vs Embedding Divergence by Session', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, "repr_vs_emb_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Representation trajectory analysis for open-weight LLM debates")
    parser.add_argument("--traces_dir", required=True)
    parser.add_argument("--output_dir", default="./repr_results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print("\n=== Loading traces ===")
    traces = load_traces(args.traces_dir)
    if not traces:
        print("No traces with hidden states found.")
        sys.exit(1)

    # Analyze
    print("\n=== Computing representation trajectories ===")
    all_results = analyze_all(traces)

    # Save
    out_file = os.path.join(args.output_dir, "representation_analysis.json")

    # Serialize numpy types
    def serialize(obj):
        if isinstance(obj, (np.floating, float)):
            return None if np.isnan(obj) else round(float(obj), 6)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=serialize)
    print(f"\nResults saved to {out_file}")

    # Print cross-session summary
    cs = all_results["cross_session"]
    print(f"\n{'='*60}")
    print(f"  Sessions analyzed: {cs['n_sessions']}")
    print(f"  Mean ρ (inter-agent distance): {cs.get('mean_rho', 'N/A')}")
    print(f"  Diverging: {cs.get('diverging_count', 0)}")
    print(f"  Converging: {cs.get('converging_count', 0)}")
    print(f"  Flat: {cs.get('flat_count', 0)}")
    print(f"  Repr diverges like output: {cs.get('repr_diverges_like_output', 'N/A')}")
    print(f"{'='*60}")

    # Plots
    if args.plot:
        print("\n=== Generating plots ===")
        for sid, result in all_results["sessions"].items():
            if "rounds" in result:
                plot_representation_trajectory(result, sid, args.output_dir)
        plot_summary(cs, all_results, args.output_dir)

    print("\n=== Representation analysis complete ===")


if __name__ == "__main__":
    main()
