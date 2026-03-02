#!/usr/bin/env python3
"""
pid_analysis.py — Partial Information Decomposition for Multi-Agent Debate Traces
Paper A, Hypothesis H4: PID–Entropy Convergent Validity

Computes synergy, redundancy, and unique information trajectories over debate rounds
using Williams & Beer (2010) PID decomposition via the `dit` library.

Correlates PID redundancy trajectory with embedding entropy H(t) trajectory to test
convergent validity (H4: r > 0.5).

Requirements:
    pip install dit numpy scikit-learn hdbscan sentence-transformers scipy matplotlib

    Note: `dit` depends on `pycddlib` which requires C build tools:
        apt-get install build-essential libgmp-dev  # Ubuntu/Debian
        # or: yum install gcc gmp-devel             # RHEL/CentOS

Usage:
    python pid_analysis.py --traces_dir ./debate_traces --output_dir ./pid_results
    python pid_analysis.py --traces_dir ./debate_traces --entropy_file ./entropy_results.json --test_h4

Author: Herald (Paper A collaboration with Chaewon)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

try:
    import dit
    from dit.pid import PID_WB  # Williams & Beer decomposition
    HAS_DIT = True
except ImportError:
    HAS_DIT = False
    print("WARNING: `dit` library not found. Install with: pip install dit")
    print("PID computation will be unavailable. Install dit and rerun.")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("WARNING: `hdbscan` not found. Falling back to k-means for clustering.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("WARNING: `sentence-transformers` not found. Provide pre-computed embeddings.")

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
    """
    Load debate traces from JSON files.

    Expected format per file:
    {
        "session_id": "startup_ideation_01",
        "task_type": "startup_ideation",
        "communication_condition": "natural_sequential",
        "agent_configs": [
            {"agent_id": "agent_0", "system_prompt": "...", "temperature": 0.7},
            {"agent_id": "agent_1", "system_prompt": "...", "temperature": 0.9}
        ],
        "rounds": [
            {
                "round": 0,
                "turns": [
                    {"agent_id": "agent_0", "text": "...", "embedding": [0.1, 0.2, ...]},
                    {"agent_id": "agent_1", "text": "...", "embedding": [0.3, 0.4, ...]}
                ]
            },
            ...
        ]
    }

    If embeddings are not pre-computed, they will be generated using sentence-transformers.
    """
    traces = []
    traces_path = Path(traces_dir)

    for f in sorted(traces_path.glob("*.json")):
        with open(f) as fh:
            trace = json.load(fh)
            traces.append(trace)

    print(f"Loaded {len(traces)} debate traces from {traces_dir}")
    return traces


def compute_embeddings_if_needed(traces: list[dict], model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """Embed turn texts if embeddings not already present."""
    # Check if embeddings exist
    sample_turn = traces[0]["rounds"][0]["turns"][0]
    if "embedding" in sample_turn and sample_turn["embedding"] is not None:
        print("Embeddings already present in traces, skipping.")
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
# 2. CLUSTERING (Discretization for PID)
# ============================================================

def cluster_embeddings(traces: list[dict], min_cluster_size: int = 3) -> tuple[list[dict], dict]:
    """
    Cluster all turn embeddings across all traces to create discrete random variables
    for PID computation.

    Uses HDBSCAN (preferred — no need to specify k, noise points = genuinely novel ideas)
    with fallback to k-means.

    Returns:
        traces: updated with 'cluster_id' per turn
        cluster_info: metadata about clustering
    """
    # Collect all embeddings
    all_embeddings = []
    embedding_index = []  # (trace_idx, round_idx, turn_idx)

    for t_idx, trace in enumerate(traces):
        for r_idx, round_data in enumerate(trace["rounds"]):
            for turn_idx, turn in enumerate(round_data["turns"]):
                all_embeddings.append(turn["embedding"])
                embedding_index.append((t_idx, r_idx, turn_idx))

    X = np.array(all_embeddings)
    print(f"Clustering {X.shape[0]} turn embeddings (dim={X.shape[1]})...")

    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='cosine')
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        method = "HDBSCAN"
        print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

        # Remap noise points to their own cluster (PID needs discrete values, not -1)
        # Each noise point gets a unique cluster to preserve information
        next_id = labels.max() + 1
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = next_id
                next_id += 1
    else:
        from sklearn.cluster import KMeans
        # Heuristic: sqrt(n) clusters, capped
        k = min(int(np.sqrt(X.shape[0])), 20)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        n_clusters = k
        n_noise = 0
        method = f"KMeans(k={k})"
        print(f"KMeans: {n_clusters} clusters")

    # Assign cluster IDs back to traces
    for idx, (t_idx, r_idx, turn_idx) in enumerate(embedding_index):
        traces[t_idx]["rounds"][r_idx]["turns"][turn_idx]["cluster_id"] = int(labels[idx])

    cluster_info = {
        "method": method,
        "n_clusters": n_clusters,
        "n_noise": n_noise if HAS_HDBSCAN else 0,
        "n_total": X.shape[0],
    }

    return traces, cluster_info


# ============================================================
# 3. PID COMPUTATION
# ============================================================

def compute_pid_for_round(agent_clusters: dict[str, list[int]], target_round: int) -> dict:
    """
    Compute PID for a single round using TIME-DELAYED mutual information (Riedl framing).

    Sources: Agent 0's cluster at round t, Agent 1's cluster at round t
    Target: The conversational "next state" — the cluster assignment at round t+1.

    This measures: how much of WHERE THE CONVERSATION GOES NEXT is predicted by
    each agent's current contribution (uniquely, redundantly, or synergistically).

    Key interpretation:
    - High synergy: both agents needed to predict next state → productive collaboration
    - High redundancy: either agent alone predicts next state → convergence / echo
    - High unique_0: agent 0 drives the conversation → anchoring
    - Declining synergy over rounds → agents becoming substitutable → convergence signal

    NOTE: We use a SLIDING WINDOW of rounds [0..target_round] to build the empirical
    distribution, giving us enough samples for a meaningful distribution even at early rounds.

    Returns dict with: synergy, redundancy, unique_0, unique_1
    """
    if not HAS_DIT:
        return {"synergy": np.nan, "redundancy": np.nan,
                "unique_0": np.nan, "unique_1": np.nan, "error": "dit not installed"}

    agents = sorted(agent_clusters.keys())
    if len(agents) < 2:
        return {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0}

    # For 2 agents: build joint distribution P(X_t, Y_t, Z_{t+1})
    # X_t = agent 0 cluster at round t
    # Y_t = agent 1 cluster at round t
    # Z_{t+1} = next round's conversational state (first agent's cluster at t+1,
    #           or combined next-round state)
    # We use all rounds [0..target_round-1] as samples (each gives one (X, Y, Z) triple)

    a0_clusters = agent_clusters[agents[0]]
    a1_clusters = agent_clusters[agents[1]]

    # Need at least target_round+1 entries to have a "next state" for round target_round-1
    n = min(len(a0_clusters), len(a1_clusters))
    max_t = min(target_round, n - 2)  # -2 because we need t+1 to exist

    if max_t < 0:
        return {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0,
                "note": "insufficient_rounds_for_time_delay"}

    # Build empirical joint distribution from all rounds [0..max_t]
    # Each round t contributes: (X=a0[t], Y=a1[t], Z=next_state[t+1])
    # For Z (next state), we use the combined cluster of both agents at t+1
    # This is NOT deterministic given X_t, Y_t — it depends on the conversation dynamics
    outcomes = {}
    for t in range(max_t + 1):
        x = str(a0_clusters[t])
        y = str(a1_clusters[t])
        # Next state: combined cluster assignment at t+1
        z = f"{a0_clusters[t+1]}_{a1_clusters[t+1]}"
        key = (x, y, z)
        outcomes[key] = outcomes.get(key, 0) + 1

    if len(outcomes) < 2:
        # Degenerate distribution — all transitions identical
        return {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0,
                "note": "degenerate_distribution"}

    # Normalize to probabilities
    total = sum(outcomes.values())
    outcome_list = list(outcomes.keys())
    probs = [outcomes[k] / total for k in outcome_list]

    try:
        # Create dit distribution
        d = dit.Distribution(outcome_list, probs)
        d.set_rv_names(['X', 'Y', 'Z'])

        # Compute PID using Williams & Beer
        pid = PID_WB(d, ['X', 'Y'], 'Z')

        result = {
            "synergy": float(pid.get_partial(((0, 1),))),
            "redundancy": float(pid.get_partial(((0,), (1,)))),
            "unique_0": float(pid.get_partial(((0,),))),
            "unique_1": float(pid.get_partial(((1,),))),
        }
    except Exception as e:
        result = {
            "synergy": np.nan, "redundancy": np.nan,
            "unique_0": np.nan, "unique_1": np.nan,
            "error": str(e)
        }

    return result


def compute_pid_trajectory(trace: dict) -> list[dict]:
    """
    Compute PID at each round for a single debate trace.

    Returns list of per-round PID values with trajectory metadata.
    """
    # Extract per-agent cluster sequences
    agent_clusters = {}
    for round_data in trace["rounds"]:
        for turn in round_data["turns"]:
            aid = turn["agent_id"]
            if aid not in agent_clusters:
                agent_clusters[aid] = []
            agent_clusters[aid].append(turn["cluster_id"])

    n_rounds = len(trace["rounds"])
    trajectory = []

    for r in range(n_rounds):
        pid = compute_pid_for_round(agent_clusters, r)
        pid["round"] = r
        pid["session_id"] = trace["session_id"]
        trajectory.append(pid)

    return trajectory


def compute_all_pid_trajectories(traces: list[dict]) -> dict:
    """Compute PID trajectories for all traces."""
    results = {}
    for trace in traces:
        sid = trace["session_id"]
        print(f"  Computing PID for {sid}...")
        results[sid] = compute_pid_trajectory(trace)
    return results


# ============================================================
# 4. H4 TESTING — PID–Entropy Convergent Validity
# ============================================================

def load_entropy_results(entropy_file: str) -> dict:
    """
    Load entropy H(t) results from measure_entropy.py output.

    Expected format:
    {
        "session_id": {
            "H_trajectory": [H_0, H_1, ...],
            "delta_trajectories": {"agent_0": [d_0, d_1, ...], ...}
        }
    }
    """
    with open(entropy_file) as f:
        return json.load(f)


def test_h4(pid_results: dict, entropy_results: dict) -> dict:
    """
    Test H4: PID redundancy increase correlates with embedding entropy decrease.

    For each session:
    - Compute slope of redundancy trajectory (expect positive = increasing)
    - Compute slope of H(t) trajectory (expect negative = decreasing)
    - Correlate across sessions: redundancy slope vs H(t) slope

    H4 predicts: negative correlation (r < -0.5), i.e., redundancy goes up as entropy goes down.
    """
    redundancy_slopes = []
    entropy_slopes = []
    synergy_slopes = []
    session_ids = []

    for sid, pid_traj in pid_results.items():
        if sid not in entropy_results:
            print(f"  Skipping {sid} — no entropy data")
            continue

        # Extract redundancy trajectory
        rounds = [p["round"] for p in pid_traj if not np.isnan(p.get("redundancy", np.nan))]
        redundancies = [p["redundancy"] for p in pid_traj if not np.isnan(p.get("redundancy", np.nan))]
        synergies = [p["synergy"] for p in pid_traj if not np.isnan(p.get("synergy", np.nan))]

        if len(rounds) < 3:
            continue

        # Compute slopes via linear regression
        r_slope, _, _, _, _ = stats.linregress(rounds, redundancies)
        s_slope, _, _, _, _ = stats.linregress(rounds, synergies)

        # Entropy trajectory
        H_traj = entropy_results[sid]["H_trajectory"]
        h_rounds = list(range(len(H_traj)))
        if len(h_rounds) < 3:
            continue
        h_slope, _, _, _, _ = stats.linregress(h_rounds, H_traj)

        redundancy_slopes.append(r_slope)
        synergy_slopes.append(s_slope)
        entropy_slopes.append(h_slope)
        session_ids.append(sid)

    if len(redundancy_slopes) < 5:
        return {
            "status": "insufficient_data",
            "n_sessions": len(redundancy_slopes),
            "message": "Need at least 5 sessions with both PID and entropy data"
        }

    # Core H4 test: correlation between redundancy slope and entropy slope
    r_corr, r_p = stats.pearsonr(redundancy_slopes, entropy_slopes)
    s_corr, s_p = stats.spearmanr(redundancy_slopes, entropy_slopes)

    # Secondary: synergy slope vs delta slope (if available)
    syn_corr, syn_p = stats.pearsonr(synergy_slopes, entropy_slopes)

    result = {
        "status": "complete",
        "n_sessions": len(redundancy_slopes),
        "h4_redundancy_entropy_pearson_r": round(r_corr, 4),
        "h4_redundancy_entropy_pearson_p": round(r_p, 6),
        "h4_redundancy_entropy_spearman_r": round(s_corr, 4),
        "h4_redundancy_entropy_spearman_p": round(s_p, 6),
        "h4_supported": r_corr < -0.5 and r_p < 0.05,
        "synergy_entropy_pearson_r": round(syn_corr, 4),
        "synergy_entropy_pearson_p": round(syn_p, 6),
        "interpretation": (
            "H4 SUPPORTED" if (r_corr < -0.5 and r_p < 0.05) else
            "H4 NOT SUPPORTED — redundancy and entropy trajectories do not show "
            "the predicted negative correlation (r < -0.5)"
        ),
        "detail": {
            "redundancy_slopes": redundancy_slopes,
            "entropy_slopes": entropy_slopes,
            "synergy_slopes": synergy_slopes,
            "session_ids": session_ids,
        }
    }

    return result


# ============================================================
# 5. VISUALIZATION
# ============================================================

def plot_pid_trajectory(pid_traj: list[dict], session_id: str, output_dir: str):
    """Plot synergy, redundancy, unique info trajectories for one session."""
    if not HAS_PLT:
        return

    rounds = [p["round"] for p in pid_traj]
    synergy = [p.get("synergy", 0) for p in pid_traj]
    redundancy = [p.get("redundancy", 0) for p in pid_traj]
    unique_0 = [p.get("unique_0", 0) for p in pid_traj]
    unique_1 = [p.get("unique_1", 0) for p in pid_traj]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, synergy, 'b-o', label='Synergy', linewidth=2)
    ax.plot(rounds, redundancy, 'r-s', label='Redundancy', linewidth=2)
    ax.plot(rounds, unique_0, 'g--^', label='Unique (Agent 0)', alpha=0.7)
    ax.plot(rounds, unique_1, 'm--v', label='Unique (Agent 1)', alpha=0.7)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Information (bits)', fontsize=12)
    ax.set_title(f'PID Trajectory — {session_id}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"pid_{session_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def plot_h4_correlation(h4_results: dict, output_dir: str):
    """Plot redundancy slope vs entropy slope scatter with regression line."""
    if not HAS_PLT or h4_results["status"] != "complete":
        return

    detail = h4_results["detail"]
    r_slopes = detail["redundancy_slopes"]
    h_slopes = detail["entropy_slopes"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_slopes, h_slopes, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)

    # Regression line
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
    print(f"  Saved H4 plot: {out_path}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PID Analysis for Multi-Agent Debate Traces")
    parser.add_argument("--traces_dir", required=True, help="Directory containing debate trace JSON files")
    parser.add_argument("--output_dir", default="./pid_results", help="Output directory for results")
    parser.add_argument("--entropy_file", default=None, help="Path to entropy results JSON (for H4 test)")
    parser.add_argument("--test_h4", action="store_true", help="Run H4 convergent validity test")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence embedding model")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="HDBSCAN min cluster size")
    parser.add_argument("--plot", action="store_true", help="Generate trajectory plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not HAS_DIT:
        print("\nERROR: `dit` library is required for PID computation.")
        print("Install: pip install dit")
        print("Requires build tools: apt-get install build-essential libgmp-dev")
        sys.exit(1)

    # Step 1: Load traces
    print("\n=== Step 1: Loading debate traces ===")
    traces = load_debate_traces(args.traces_dir)

    # Step 2: Compute embeddings if needed
    print("\n=== Step 2: Checking embeddings ===")
    traces = compute_embeddings_if_needed(traces, args.embedding_model)

    # Step 3: Cluster embeddings
    print("\n=== Step 3: Clustering embeddings for PID discretization ===")
    traces, cluster_info = cluster_embeddings(traces, args.min_cluster_size)
    print(f"  Clustering: {cluster_info}")

    # Step 4: Compute PID trajectories
    print("\n=== Step 4: Computing PID trajectories ===")
    pid_results = compute_all_pid_trajectories(traces)

    # Save results
    results_file = os.path.join(args.output_dir, "pid_trajectories.json")
    # Convert for JSON serialization
    serializable = {}
    for sid, traj in pid_results.items():
        serializable[sid] = []
        for entry in traj:
            clean = {k: (float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else v)
                     for k, v in entry.items()}
            serializable[sid].append(clean)

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nPID trajectories saved to {results_file}")

    # Save cluster info
    with open(os.path.join(args.output_dir, "cluster_info.json"), 'w') as f:
        json.dump(cluster_info, f, indent=2)

    # Step 5: Plots
    if args.plot:
        print("\n=== Step 5: Generating plots ===")
        for sid, traj in pid_results.items():
            plot_pid_trajectory(traj, sid, args.output_dir)

    # Step 6: H4 test
    if args.test_h4:
        if args.entropy_file is None:
            print("\nWARNING: --test_h4 requires --entropy_file. Skipping H4 test.")
        else:
            print("\n=== Step 6: Testing H4 (PID–Entropy Convergent Validity) ===")
            entropy_results = load_entropy_results(args.entropy_file)
            h4_results = test_h4(pid_results, entropy_results)

            h4_file = os.path.join(args.output_dir, "h4_results.json")
            with open(h4_file, 'w') as f:
                json.dump(h4_results, f, indent=2, default=str)
            print(f"\nH4 results saved to {h4_file}")
            print(f"\n{'='*60}")
            print(f"  H4 RESULT: {h4_results['interpretation']}")
            print(f"  Pearson r = {h4_results['h4_redundancy_entropy_pearson_r']}")
            print(f"  p-value   = {h4_results['h4_redundancy_entropy_pearson_p']}")
            print(f"  N sessions = {h4_results['n_sessions']}")
            print(f"{'='*60}")

            if args.plot:
                plot_h4_correlation(h4_results, args.output_dir)

    print("\n✅ PID analysis complete.")


if __name__ == "__main__":
    main()
