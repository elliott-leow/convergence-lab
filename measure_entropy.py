#!/usr/bin/env python3
"""
measure_entropy.py — Embedding entropy H(t) and per-agent δ(t) for debate traces
Paper A, Hypotheses H1 (compression) and H2 (critical slowing down)

Computes:
- H(t): Shannon entropy over embedding distribution at each round
- δ(t): Per-agent cosine distance between consecutive turns
- Terminal diversity: cluster count at final round
- Compression ratio: H(final)/H(initial)

Uses pre-computed embeddings from trace files.

Author: Chaewon (Paper A collaboration with Herald)
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def load_traces(traces_dir: str) -> list[dict]:
    traces = []
    for f in sorted(Path(traces_dir).glob("*.json")):
        with open(f) as fh:
            traces.append(json.load(fh))
    print(f"Loaded {len(traces)} traces from {traces_dir}")
    return traces


def compute_entropy_knn(embeddings: np.ndarray, k: int = 3) -> float:
    """Estimate Shannon entropy via k-NN density estimation (Kozachenko-Leonenko)."""
    n, d = embeddings.shape
    if n <= k:
        return 0.0

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(embeddings, embeddings, metric='cosine')
    
    # For each point, find k-th nearest neighbor distance (excluding self)
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)
    knn_dists = sorted_dists[:, k-1]  # k-th nearest neighbor
    
    # Kozachenko-Leonenko entropy estimator
    # H ≈ (d/n) * sum(log(knn_dist)) + log(n) + log(V_d) + euler_gamma
    # Simplified: we care about relative trajectory, not absolute value
    knn_dists = np.maximum(knn_dists, 1e-10)  # avoid log(0)
    H = d * np.mean(np.log(knn_dists)) + np.log(n)
    return float(H)


def compute_session_metrics(trace: dict) -> dict:
    """Compute H(t), δ(t), terminal diversity for one session."""
    session_id = trace["session_id"]
    rounds = trace["rounds"]
    n_rounds = len(rounds)
    
    # --- H(t) trajectory ---
    # Use mean pairwise cosine distance within each round as diversity proxy
    # More robust than KNN entropy for small sample sizes (2-3 agents per round)
    H_trajectory = []
    all_round_embeddings = []
    
    for round_data in rounds:
        embs = np.array([t["embedding"] for t in round_data["turns"]])
        all_round_embeddings.append(embs)
        
        if len(embs) >= 2:
            # Mean pairwise cosine distance = diversity measure
            from scipy.spatial.distance import pdist
            pairwise = pdist(embs, metric='cosine')
            H = float(np.mean(pairwise))
        else:
            H = 0.0
        H_trajectory.append(H)
    
    # --- δ(t) per agent ---
    agent_turns = {}  # agent_id -> list of embeddings by round
    for round_data in rounds:
        for turn in round_data["turns"]:
            aid = turn["agent_id"]
            if aid not in agent_turns:
                agent_turns[aid] = []
            agent_turns[aid].append(np.array(turn["embedding"]))
    
    delta_trajectories = {}
    for aid, embs in agent_turns.items():
        deltas = []
        for i in range(1, len(embs)):
            d = cosine_dist(embs[i-1], embs[i])
            deltas.append(float(d))
        delta_trajectories[aid] = deltas
    
    # --- Terminal diversity (HDBSCAN cluster count at final round) ---
    final_embs = all_round_embeddings[-1]
    if HAS_HDBSCAN and len(final_embs) >= 3:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine')
        labels = clusterer.fit_predict(final_embs)
        terminal_diversity = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        terminal_diversity = len(final_embs)  # each turn is its own "cluster"
    
    # --- Compression ratio ---
    if H_trajectory[0] != 0:
        compression_ratio = H_trajectory[-1] / H_trajectory[0]
    else:
        compression_ratio = 1.0
    
    # --- H2 test: temporal gap between δ decline and H decline ---
    # Find round where mean δ first drops below 1 SD of initial
    mean_deltas = []
    for r in range(n_rounds - 1):
        round_deltas = [delta_trajectories[aid][r] for aid in delta_trajectories if r < len(delta_trajectories[aid])]
        if round_deltas:
            mean_deltas.append(np.mean(round_deltas))
    
    delta_threshold_round = None
    H_threshold_round = None
    
    # Use percentage-based thresholds (more robust than SD for noisy early rounds)
    # δ threshold: first round where mean δ drops below 50% of initial value
    # H threshold: first round where H drops below 70% of initial value
    if len(mean_deltas) >= 3:
        initial_delta = mean_deltas[0]
        delta_threshold = initial_delta * 0.5
        
        for r, d in enumerate(mean_deltas):
            if d < delta_threshold:
                delta_threshold_round = r + 1  # +1 because deltas start at round 1
                break
    
    if len(H_trajectory) >= 3:
        initial_H = H_trajectory[0]
        H_threshold = initial_H * 0.7
        
        for r, h in enumerate(H_trajectory):
            if r == 0:
                continue  # skip first round
            if h < H_threshold:
                H_threshold_round = r
                break
    
    temporal_gap = None
    if delta_threshold_round is not None and H_threshold_round is not None:
        temporal_gap = H_threshold_round - delta_threshold_round
    
    return {
        "session_id": session_id,
        "scenario": trace.get("metadata", {}).get("scenario", "unknown"),
        "H_trajectory": H_trajectory,
        "delta_trajectories": {aid: deltas for aid, deltas in delta_trajectories.items()},
        "mean_delta_trajectory": mean_deltas,
        "terminal_diversity": terminal_diversity,
        "compression_ratio": round(compression_ratio, 4),
        "h2_analysis": {
            "delta_threshold_round": delta_threshold_round,
            "H_threshold_round": H_threshold_round,
            "temporal_gap": temporal_gap,
            "h2_supported": temporal_gap is not None and temporal_gap >= 2,
            "interpretation": (
                f"δ declined at round {delta_threshold_round}, H declined at round {H_threshold_round}, "
                f"gap = {temporal_gap} rounds"
                if temporal_gap is not None
                else "Could not determine threshold crossings"
            )
        }
    }


def plot_session(metrics: dict, output_dir: str):
    """Plot H(t) and δ(t) trajectories for one session."""
    if not HAS_PLT:
        return
    
    sid = metrics["session_id"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # H(t)
    rounds = list(range(len(metrics["H_trajectory"])))
    ax1.plot(rounds, metrics["H_trajectory"], 'b-o', linewidth=2, label='H(t)')
    ax1.set_ylabel('Entropy H(t)', fontsize=12)
    ax1.set_title(f'{sid} — Entropy & Semantic Delta', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark H threshold crossing
    h2 = metrics["h2_analysis"]
    if h2["H_threshold_round"] is not None:
        ax1.axvline(x=h2["H_threshold_round"], color='red', linestyle='--', alpha=0.5, label='H threshold')
    
    # δ(t)
    for aid, deltas in metrics["delta_trajectories"].items():
        ax2.plot(range(1, len(deltas)+1), deltas, '-o', linewidth=2, label=f'δ({aid})', alpha=0.8)
    
    if metrics["mean_delta_trajectory"]:
        ax2.plot(range(1, len(metrics["mean_delta_trajectory"])+1), 
                metrics["mean_delta_trajectory"], 'k-s', linewidth=2.5, label='mean δ(t)')
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Semantic Delta δ(t)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if h2["delta_threshold_round"] is not None:
        ax2.axvline(x=h2["delta_threshold_round"], color='green', linestyle='--', alpha=0.5, label='δ threshold')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"entropy_{sid}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Measure entropy and semantic delta for debate traces")
    parser.add_argument("--traces_dir", required=True)
    parser.add_argument("--output_dir", default="./entropy_results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    traces = load_traces(args.traces_dir)
    
    all_results = {}
    for trace in traces:
        sid = trace["session_id"]
        print(f"\nProcessing {sid}...")
        metrics = compute_session_metrics(trace)
        all_results[sid] = metrics
        
        print(f"  H(t): {[round(h, 3) for h in metrics['H_trajectory']]}")
        print(f"  mean δ(t): {[round(d, 4) for d in metrics['mean_delta_trajectory']]}")
        print(f"  Terminal diversity: {metrics['terminal_diversity']}")
        print(f"  Compression ratio: {metrics['compression_ratio']}")
        print(f"  H2: {metrics['h2_analysis']['interpretation']}")
        
        if args.plot:
            plot_session(metrics, args.output_dir)
    
    # Save results
    out_file = os.path.join(args.output_dir, "entropy_results.json")
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for sid, m in all_results.items():
        scenario = m["scenario"]
        h2 = m["h2_analysis"]
        print(f"\n{sid} ({scenario}):")
        print(f"  Compression ratio: {m['compression_ratio']}")
        print(f"  H2 gap: {h2['temporal_gap']} rounds")
        print(f"  H2 supported: {h2['h2_supported']}")


if __name__ == "__main__":
    main()
