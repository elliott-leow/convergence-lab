#!/usr/bin/env python3
"""
Compute embedding entropy H(t), per-agent delta δ(t), and compression metrics
across debate traces. Tests H1, H2, H5.
"""
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import entropy as scipy_entropy


def load_traces(trace_dir):
    """Load all debate traces from directory."""
    traces = []
    for f in sorted(Path(trace_dir).glob("*.json")):
        with open(f) as fh:
            traces.append(json.load(fh))
    return traces


def get_all_responses_by_round(trace):
    """Extract responses organized by round and agent."""
    rounds = []
    for r in trace["rounds"]:
        round_data = {}
        for turn in r["turns"]:
            round_data[turn["agent"]] = turn["response"]
        rounds.append(round_data)
    return rounds


def compute_embedding_entropy(embeddings):
    """
    Compute entropy of embedding distribution using PCA-based approach.
    Lower entropy = more compressed/converged semantic space.
    """
    if len(embeddings) < 2:
        return 0.0

    emb = np.array(embeddings)
    # Center
    emb = emb - emb.mean(axis=0)

    # Compute covariance eigenvalues
    n_components = min(len(embeddings), emb.shape[1], 50)
    if n_components < 2:
        return 0.0

    pca = PCA(n_components=n_components)
    pca.fit(emb)

    # Use explained variance ratios as probability distribution
    ratios = pca.explained_variance_ratio_
    ratios = ratios[ratios > 1e-10]  # filter near-zero
    if len(ratios) == 0:
        return 0.0

    return scipy_entropy(ratios)


def compute_intrinsic_dimensionality(embeddings):
    """Estimate intrinsic dimensionality via PCA (participation ratio)."""
    if len(embeddings) < 3:
        return float('nan')

    emb = np.array(embeddings)
    emb = emb - emb.mean(axis=0)

    n_components = min(len(embeddings), emb.shape[1], 50)
    pca = PCA(n_components=n_components)
    pca.fit(emb)

    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # Participation ratio: (sum λ)^2 / sum(λ^2)
    if np.sum(eigenvalues**2) == 0:
        return 0.0
    return (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)


def compute_pairwise_similarity(embeddings):
    """Average pairwise cosine similarity between embeddings."""
    if len(embeddings) < 2:
        return 0.0

    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            sims.append(sim)
    return np.mean(sims)


def compute_per_agent_delta(agent_embeddings_by_round):
    """
    Compute per-agent δ(t): semantic distance between an agent's consecutive responses.
    Low δ = agent is repeating itself / stagnating.
    """
    deltas = {}
    for agent, emb_list in agent_embeddings_by_round.items():
        agent_deltas = []
        for i in range(1, len(emb_list)):
            if emb_list[i] is not None and emb_list[i-1] is not None:
                d = cosine(emb_list[i], emb_list[i-1])
                agent_deltas.append(d)
        deltas[agent] = agent_deltas
    return deltas


def analyze_trace(trace, models):
    """Analyze a single debate trace with multiple embedding models."""
    rounds = get_all_responses_by_round(trace)
    results = {
        "task_id": trace["task_id"],
        "condition": trace["condition"],
        "num_rounds": len(rounds),
        "models": {}
    }

    for model_name, model in models.items():
        # Collect all responses and embed them
        all_responses = []
        agent_responses = defaultdict(list)
        round_responses = []

        for r in rounds:
            round_texts = []
            for agent, resp in r.items():
                all_responses.append(resp)
                agent_responses[agent].append(resp)
                round_texts.append(resp)
            round_responses.append(round_texts)

        # Embed everything at once
        all_embeddings = model.encode(all_responses, show_progress_bar=False)

        # Map back to rounds and agents
        idx = 0
        agent_embeddings_by_round = defaultdict(list)
        round_embeddings = []

        for r in rounds:
            re = []
            for agent in r:
                agent_embeddings_by_round[agent].append(all_embeddings[idx])
                re.append(all_embeddings[idx])
                idx += 1
            round_embeddings.append(re)

        # Cumulative entropy over rounds
        cumulative_embs = []
        entropy_trajectory = []
        dimensionality_trajectory = []
        similarity_trajectory = []

        for re in round_embeddings:
            cumulative_embs.extend(re)
            h = compute_embedding_entropy(cumulative_embs)
            d = compute_intrinsic_dimensionality(cumulative_embs)
            s = compute_pairwise_similarity(re)
            entropy_trajectory.append(h)
            dimensionality_trajectory.append(d)
            similarity_trajectory.append(s)

        # Per-agent deltas
        deltas = compute_per_agent_delta(agent_embeddings_by_round)

        results["models"][model_name] = {
            "entropy_trajectory": entropy_trajectory,
            "dimensionality_trajectory": dimensionality_trajectory,
            "similarity_trajectory": similarity_trajectory,
            "per_agent_delta": {a: d for a, d in deltas.items()},
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default="./traces")
    parser.add_argument("--output", default="./analysis/entropy_results.json")
    parser.add_argument("--models", nargs="+",
                        default=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"])
    parser.add_argument("--include-bow", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    # Load embedding models
    print("Loading embedding models...")
    models = {}
    for m in args.models:
        print(f"  Loading {m}...")
        models[m] = SentenceTransformer(m, device="cuda")

    # Load traces
    traces = load_traces(args.trace_dir)
    print(f"Loaded {len(traces)} traces")

    # Analyze each trace
    all_results = []
    for i, trace in enumerate(traces):
        print(f"\nAnalyzing trace {i+1}/{len(traces)}: {trace['task_id']} ({trace['condition']})")
        result = analyze_trace(trace, models)

        # BoW baseline (H5)
        if args.include_bow:
            rounds = get_all_responses_by_round(trace)
            all_texts = []
            for r in rounds:
                all_texts.extend(r.values())

            vectorizer = CountVectorizer(max_features=5000)
            bow_matrix = vectorizer.fit_transform(all_texts).toarray()

            idx = 0
            cumulative = []
            bow_entropy = []
            for r in rounds:
                for _ in r:
                    cumulative.append(bow_matrix[idx])
                    idx += 1
                h = compute_embedding_entropy(cumulative)
                bow_entropy.append(h)

            result["models"]["bow_baseline"] = {
                "entropy_trajectory": bow_entropy,
                "dimensionality_trajectory": [],
                "similarity_trajectory": [],
                "per_agent_delta": {}
            }

        all_results.append(result)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else None)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    import os
    main()
