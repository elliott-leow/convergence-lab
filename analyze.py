#!/usr/bin/env python3
"""
Statistical analysis and hypothesis testing for Paper A.
Reads entropy results and self-consistency data, tests H1-H5.
"""
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict


def test_h1_monotonic_decrease(results):
    """H1: Embedding entropy H(t) decreases monotonically over rounds."""
    print("\n" + "="*60)
    print("H1: Monotonic entropy decrease")
    print("="*60)

    for model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]:
        trajectories = []
        for r in results:
            if model_name in r.get("models", {}):
                traj = r["models"][model_name]["entropy_trajectory"]
                if len(traj) > 1:
                    trajectories.append(traj)

        if not trajectories:
            continue

        # Test: is average trajectory monotonically decreasing?
        avg_traj = np.mean(trajectories, axis=0)
        diffs = np.diff(avg_traj)
        n_decreasing = np.sum(diffs < 0)
        n_total = len(diffs)

        # Spearman correlation between round number and entropy
        all_correlations = []
        for traj in trajectories:
            r_val, p_val = stats.spearmanr(range(len(traj)), traj)
            all_correlations.append(r_val)

        mean_corr = np.mean(all_correlations)
        # One-sample t-test: is mean correlation significantly negative?
        t_stat, p_val = stats.ttest_1samp(all_correlations, 0)

        print(f"\n  Model: {model_name}")
        print(f"  N sessions: {len(trajectories)}")
        print(f"  Avg trajectory decreasing steps: {n_decreasing}/{n_total}")
        print(f"  Mean Spearman r(round, entropy): {mean_corr:.4f}")
        print(f"  t-test vs 0: t={t_stat:.3f}, p={p_val:.6f}")
        print(f"  H1 {'SUPPORTED' if mean_corr < 0 and p_val < 0.05 else 'NOT SUPPORTED'}")


def test_h2_leading_indicator(results):
    """H2: Per-agent δ(t) declines before group H(t)."""
    print("\n" + "="*60)
    print("H2: δ(t) as leading indicator of H(t) collapse")
    print("="*60)

    model_name = "all-MiniLM-L6-v2"
    lead_times = []

    for r in results:
        if model_name not in r.get("models", {}):
            continue

        m = r["models"][model_name]
        entropy = np.array(m["entropy_trajectory"])
        deltas = m.get("per_agent_delta", {})

        if len(entropy) < 4 or not deltas:
            continue

        # Find when entropy drops below 1 SD of initial
        if len(entropy) < 3:
            continue
        h_threshold = entropy[0] - np.std(entropy[:3])
        h_cross = None
        for i, h in enumerate(entropy):
            if h < h_threshold:
                h_cross = i
                break

        # Find when avg delta drops below 1 SD of initial
        all_deltas = []
        for agent, d in deltas.items():
            all_deltas.append(d)

        if not all_deltas or not all_deltas[0]:
            continue

        avg_delta = np.mean(all_deltas, axis=0)
        if len(avg_delta) < 3:
            continue

        d_threshold = avg_delta[0] - np.std(avg_delta[:3])
        d_cross = None
        for i, d in enumerate(avg_delta):
            if d < d_threshold:
                d_cross = i
                break

        if h_cross is not None and d_cross is not None:
            lead_time = h_cross - d_cross
            lead_times.append(lead_time)

    if lead_times:
        mean_lead = np.mean(lead_times)
        t_stat, p_val = stats.ttest_1samp(lead_times, 0)
        print(f"  N sessions with both crossings: {len(lead_times)}")
        print(f"  Mean lead time (rounds): {mean_lead:.2f}")
        print(f"  t-test vs 0: t={t_stat:.3f}, p={p_val:.6f}")
        print(f"  H2 {'SUPPORTED' if mean_lead > 0 and p_val < 0.05 else 'NOT SUPPORTED'}")
        print(f"  (positive = δ crosses first, as predicted)")
    else:
        print("  Insufficient data for H2 test")


def test_h3_inverted_u(consistency_data, debate_results):
    """H3: Inverted-U between self-consistency and debate quality."""
    print("\n" + "="*60)
    print("H3: Self-consistency inverted-U with debate quality")
    print("="*60)

    if not consistency_data:
        print("  No self-consistency data available")
        return

    # Extract consistency scores per agent per task
    for agent_id, tasks in consistency_data.items():
        scores = []
        for task_id, data in tasks.items():
            scores.append(data["consistency"]["consistency_score"])

        print(f"\n  {agent_id}:")
        print(f"    Mean consistency: {np.mean(scores):.4f}")
        print(f"    Std consistency: {np.std(scores):.4f}")
        print(f"    Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")

    print("\n  Note: Full inverted-U test requires matching consistency scores")
    print("  with debate quality metrics from varied agent configurations.")
    print("  This requires more than 2 agents — needs multiple OLMo configs.")


def test_h5_bow_robustness(results):
    """H5: Compression absent in BoW baseline."""
    print("\n" + "="*60)
    print("H5: Embedding robustness (neural vs BoW)")
    print("="*60)

    neural_corrs = defaultdict(list)
    bow_corrs = []

    for r in results:
        for model_name, m in r.get("models", {}).items():
            traj = m["entropy_trajectory"]
            if len(traj) < 3:
                continue
            rho, _ = stats.spearmanr(range(len(traj)), traj)
            if model_name == "bow_baseline":
                bow_corrs.append(rho)
            else:
                neural_corrs[model_name].append(rho)

    print("\n  Neural embedder correlations (round vs entropy):")
    for model, corrs in neural_corrs.items():
        print(f"    {model}: mean ρ = {np.mean(corrs):.4f} (n={len(corrs)})")

    if bow_corrs:
        print(f"\n  BoW baseline: mean ρ = {np.mean(bow_corrs):.4f} (n={len(bow_corrs)})")

        # Test: are neural correlations significantly more negative than BoW?
        all_neural = []
        for corrs in neural_corrs.values():
            all_neural.extend(corrs)

        if all_neural and bow_corrs:
            t_stat, p_val = stats.ttest_ind(all_neural, bow_corrs)
            print(f"\n  Neural vs BoW t-test: t={t_stat:.3f}, p={p_val:.6f}")
            print(f"  H5 {'SUPPORTED' if np.mean(all_neural) < np.mean(bow_corrs) and p_val < 0.05 else 'NOT SUPPORTED'}")
    else:
        print("  No BoW data available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy-results", default="./analysis/entropy_results.json")
    parser.add_argument("--consistency-results", default="./analysis/self_consistency.json")
    parser.add_argument("--output", default="./analysis/hypothesis_tests.txt")
    args = parser.parse_args()

    # Load data
    entropy_results = []
    if Path(args.entropy_results).exists():
        with open(args.entropy_results) as f:
            entropy_results = json.load(f)

    consistency_data = {}
    if Path(args.consistency_results).exists():
        with open(args.consistency_results) as f:
            consistency_data = json.load(f)

    # Run tests
    import sys
    from io import StringIO
    output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = output

    print("Paper A: Hypothesis Test Results")
    print(f"N debate sessions: {len(entropy_results)}")
    print(f"Has consistency data: {bool(consistency_data)}")

    test_h1_monotonic_decrease(entropy_results)
    test_h2_leading_indicator(entropy_results)
    test_h3_inverted_u(consistency_data, entropy_results)
    test_h5_bow_robustness(entropy_results)

    sys.stdout = old_stdout
    report = output.getvalue()
    print(report)

    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
