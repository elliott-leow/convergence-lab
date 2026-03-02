#!/usr/bin/env python3
"""
pid_analysis_v3.py — PID via Embedding Delta Mutual Information
Paper A, Hypothesis H4: PID–Entropy Convergent Validity

v3: Drops the convergence/divergence assumption from v2's binary discretization.
Instead of asking "did agent move toward or away?", we ask:
"does knowing agent A's change help predict agent B's change?"

Key insight (from v2 failure analysis):
    Agents can be COUPLED without CONVERGING. Two scouts reading each other's
    reports and fanning out to different areas are coordinating — but a "toward/away"
    metric codes them as independent. Delta-MI captures coupling regardless of
    direction.

Method:
    1. Compute embedding deltas: Δ_i(t) = emb_i(t) - emb_i(t-1) for each agent
    2. Project deltas to low-dimensional space (PCA) for tractable discretization
    3. Discretize projected deltas into bins (quantile-based)
    4. Compute PID on discretized deltas:
       - Sources: agent_0's binned delta, agent_1's binned delta at round t
       - Target: combined next-round delta bins at round t+1
    5. Interpret:
       - High synergy = agents jointly determine conversation trajectory
       - High redundancy = agents make interchangeable moves (convergence)
       - High unique = one agent drives while other follows
       - MI > 0 at all = agents are coupled (not parallel monologues)

    Also computes raw MI(Δa, Δb) per round as a simpler coupling measure.

Requirements:
    pip install dit numpy scikit-learn sentence-transformers scipy matplotlib

Usage:
    python pid_analysis_v3.py --traces_dir ./debate_traces --output_dir ./pid_results --plot
    python pid_analysis_v3.py --traces_dir ./traces --output_dir ./pid_results --pool --plot
    python pid_analysis_v3.py --traces_dir ./traces --entropy_file ./entropy.json --test_h4 --plot

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
    import dit
    from dit.pid import PID_WB
    HAS_DIT = True
except ImportError:
    HAS_DIT = False
    print("WARNING: `dit` not found. Install: pip install dit")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not found, PCA projection unavailable")

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
    for f in sorted(Path(traces_dir).glob("*.json")):
        with open(f) as fh:
            traces.append(json.load(fh))
    print(f"Loaded {len(traces)} traces from {traces_dir}")
    return traces


def compute_embeddings_if_needed(traces: list[dict], model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """Embed turn texts if not already present."""
    sample = traces[0]["rounds"][0]["turns"][0]
    if "embedding" in sample and sample["embedding"] is not None:
        print("Embeddings present.")
        return traces
    if not HAS_ST:
        raise RuntimeError("No embeddings and sentence-transformers not installed.")
    print(f"Computing embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    for trace in traces:
        for rd in trace["rounds"]:
            texts = [t["text"] for t in rd["turns"]]
            embs = model.encode(texts)
            for turn, emb in zip(rd["turns"], embs):
                turn["embedding"] = emb.tolist()
    return traces


# ============================================================
# 2. EMBEDDING DELTAS
# ============================================================

def extract_agent_embeddings(trace: dict) -> dict[str, list[np.ndarray]]:
    """Extract per-agent embedding sequences from a trace."""
    agent_embs = {}
    for rd in trace["rounds"]:
        for turn in rd["turns"]:
            aid = turn["agent_id"]
            if aid not in agent_embs:
                agent_embs[aid] = []
            agent_embs[aid].append(np.array(turn["embedding"]))
    return agent_embs


def compute_deltas(agent_embs: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    """
    Compute embedding deltas: delta_i(t) = emb_i(t) - emb_i(t-1).
    Returns dict mapping agent_id -> list of delta vectors (length n_rounds - 1).
    """
    deltas = {}
    for aid, embs in agent_embs.items():
        deltas[aid] = [embs[t] - embs[t-1] for t in range(1, len(embs))]
    return deltas


# ============================================================
# 3. RAW COUPLING: MI(delta_a, delta_b) PER ROUND
# ============================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, handling zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_delta_coupling(trace: dict) -> list[dict]:
    """
    Compute raw coupling metrics between agent deltas per round.

    For each round t (starting from round 2, since deltas start at round 1):
    - cosine_sim(delta_a(t), delta_b(t)): are they moving in the same direction?
    - cosine_sim(delta_a(t), delta_b(t-1)): does agent A's move follow B's previous?
    - magnitude ratio: are they moving by similar amounts?

    These are continuous metrics (no discretization needed) that directly measure
    coupling without imposing a convergence frame.
    """
    agent_embs = extract_agent_embeddings(trace)
    agents = sorted(agent_embs.keys())
    if len(agents) < 2:
        return []

    deltas = compute_deltas(agent_embs)
    a0_deltas = deltas[agents[0]]
    a1_deltas = deltas[agents[1]]
    n = min(len(a0_deltas), len(a1_deltas))

    coupling = []
    for t in range(n):
        d0 = a0_deltas[t]
        d1 = a1_deltas[t]

        # Simultaneous coupling: are they moving in the same direction right now?
        sim_simultaneous = cosine_sim(d0, d1)

        # Lagged coupling: does A's current move correlate with B's previous?
        sim_a_leads = cosine_sim(d0, a1_deltas[t-1]) if t > 0 else 0.0
        sim_b_leads = cosine_sim(d1, a0_deltas[t-1]) if t > 0 else 0.0

        # Magnitude coupling: are they moving similar distances?
        mag_0 = float(np.linalg.norm(d0))
        mag_1 = float(np.linalg.norm(d1))
        mag_ratio = min(mag_0, mag_1) / max(mag_0, mag_1) if max(mag_0, mag_1) > 1e-10 else 1.0

        # Cumulative coupling: running average of simultaneous cosine similarity
        # If this is consistently > 0, agents are systematically coupled
        all_sims = [cosine_sim(a0_deltas[i], a1_deltas[i]) for i in range(t+1)]
        mean_coupling = float(np.mean(all_sims))

        coupling.append({
            "round": t + 2,  # deltas start at round 1, coupling at round 2
            "session_id": trace.get("session_id", "unknown"),
            "delta_cosine_simultaneous": round(sim_simultaneous, 6),
            "delta_cosine_a_leads": round(sim_a_leads, 6),
            "delta_cosine_b_leads": round(sim_b_leads, 6),
            "magnitude_ratio": round(mag_ratio, 6),
            "magnitude_agent_0": round(mag_0, 6),
            "magnitude_agent_1": round(mag_1, 6),
            "cumulative_mean_coupling": round(mean_coupling, 6),
        })

    return coupling


# ============================================================
# 4. PID ON DISCRETIZED DELTAS
# ============================================================

def discretize_deltas_quantile(all_deltas: list[np.ndarray], n_bins: int = 3,
                                n_components: int = 2) -> list[str]:
    """
    Discretize embedding deltas for PID computation.

    1. Project to low-D via PCA (reduces state space)
    2. Quantile-bin each component independently
    3. Combine bins into a single label string

    This preserves DIRECTION information (unlike binary toward/away)
    while keeping the state space tractable.
    """
    if len(all_deltas) < n_bins:
        return [str(i) for i in range(len(all_deltas))]

    X = np.array(all_deltas)

    # PCA projection
    n_comp = min(n_components, X.shape[1], X.shape[0])
    if HAS_SKLEARN and X.shape[1] > n_comp:
        pca = PCA(n_components=n_comp)
        X_proj = pca.fit_transform(X)
    else:
        X_proj = X[:, :n_comp]

    # Quantile binning per component
    labels = []
    for i in range(X_proj.shape[0]):
        bins = []
        for j in range(X_proj.shape[1]):
            col = X_proj[:, j]
            # Which quantile does this value fall in?
            percentile = stats.percentileofscore(col, X_proj[i, j], kind='rank')
            bin_id = min(int(percentile / (100 / n_bins)), n_bins - 1)
            bins.append(str(bin_id))
        labels.append("".join(bins))

    return labels


def compute_pid_delta(traces: list[dict], n_bins: int = 3, pool: bool = False) -> dict:
    """
    Compute PID on embedding deltas.

    Sources: agent_0's discretized delta at round t, agent_1's discretized delta at t
    Target: combined discretized deltas at round t+1

    If pool=True, aggregate all transitions across traces (by condition).
    """
    if not HAS_DIT:
        return {"error": "dit not installed"}

    if pool:
        return _compute_pid_delta_pooled(traces, n_bins)
    else:
        return _compute_pid_delta_per_session(traces, n_bins)


def _compute_pid_delta_per_session(traces: list[dict], n_bins: int) -> dict:
    """Per-session PID trajectories using discretized deltas."""
    results = {}

    for trace in traces:
        sid = trace.get("session_id", "unknown")
        agent_embs = extract_agent_embeddings(trace)
        agents = sorted(agent_embs.keys())
        if len(agents) < 2:
            continue

        deltas = compute_deltas(agent_embs)
        a0_deltas = deltas[agents[0]]
        a1_deltas = deltas[agents[1]]
        n = min(len(a0_deltas), len(a1_deltas))

        if n < 4:  # need enough for discretization + transitions
            results[sid] = [{"round": t+2, "note": "insufficient_rounds"} for t in range(n)]
            continue

        # Discretize all deltas together for this session
        all_d = a0_deltas[:n] + a1_deltas[:n]
        all_labels = discretize_deltas_quantile(all_d, n_bins=n_bins)
        a0_labels = all_labels[:n]
        a1_labels = all_labels[n:]

        trajectory = []
        for t in range(n):
            # Cumulative transitions up to round t
            transitions = []
            for tp in range(min(t, n - 1)):
                x = a0_labels[tp]
                y = a1_labels[tp]
                z = f"{a0_labels[tp+1]}_{a1_labels[tp+1]}"
                transitions.append((x, y, z))

            if len(transitions) >= 3:
                pid = _run_pid(transitions)
            else:
                pid = {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0,
                       "note": "warming_up"}

            pid["round"] = t + 2
            pid["session_id"] = sid
            pid["n_transitions"] = len(transitions)
            trajectory.append(pid)

        results[sid] = trajectory
        print(f"  {sid}: {len(trajectory)} rounds, final n_transitions={trajectory[-1]['n_transitions']}")

    return results


def _compute_pid_delta_pooled(traces: list[dict], n_bins: int) -> dict:
    """Pool transitions by condition for richer distributions."""
    # First, collect all deltas across all traces for global discretization
    condition_data = defaultdict(lambda: {"a0_deltas": [], "a1_deltas": []})

    for trace in traces:
        condition = trace.get("communication_condition", "unknown")
        agent_embs = extract_agent_embeddings(trace)
        agents = sorted(agent_embs.keys())
        if len(agents) < 2:
            continue

        deltas = compute_deltas(agent_embs)
        n = min(len(deltas[agents[0]]), len(deltas[agents[1]]))
        condition_data[condition]["a0_deltas"].extend(deltas[agents[0]][:n])
        condition_data[condition]["a1_deltas"].extend(deltas[agents[1]][:n])

    results = {}
    for condition, data in condition_data.items():
        a0_d = data["a0_deltas"]
        a1_d = data["a1_deltas"]
        n = min(len(a0_d), len(a1_d))

        if n < 5:
            results[condition] = {"note": "insufficient_data", "n_deltas": n}
            continue

        # Discretize together
        all_d = a0_d + a1_d
        all_labels = discretize_deltas_quantile(all_d, n_bins=n_bins)
        a0_labels = all_labels[:n]
        a1_labels = all_labels[n:]

        # Build transitions
        transitions = []
        for t in range(n - 1):
            x = a0_labels[t]
            y = a1_labels[t]
            z = f"{a0_labels[t+1]}_{a1_labels[t+1]}"
            transitions.append((x, y, z))

        print(f"  Condition '{condition}': {len(transitions)} transitions")
        pid = _run_pid(transitions)
        pid["condition"] = condition
        pid["n_transitions"] = len(transitions)
        results[condition] = pid

    return results


def _run_pid(transitions: list[tuple]) -> dict:
    """Run Williams-Beer PID on a list of (X, Y, Z) triples."""
    if not HAS_DIT:
        return {"error": "dit not installed"}

    counts = {}
    for t in transitions:
        counts[t] = counts.get(t, 0) + 1

    total = sum(counts.values())
    outcomes = list(counts.keys())
    probs = [counts[k] / total for k in outcomes]

    if len(outcomes) < 2:
        return {"synergy": 0, "redundancy": 0, "unique_0": 0, "unique_1": 0,
                "note": "degenerate_single_outcome", "n_unique": len(outcomes)}

    try:
        d = dit.Distribution(outcomes, probs)
        d.set_rv_names(['X', 'Y', 'Z'])
        pid = PID_WB(d, ['X', 'Y'], 'Z')

        # Try both API variants (get_partial for older dit, get_pi for newer)
        try:
            syn = float(pid.get_partial(((0, 1),)))
            red = float(pid.get_partial(((0,), (1,))))
            u0 = float(pid.get_partial(((0,),)))
            u1 = float(pid.get_partial(((1,),)))
        except (AttributeError, TypeError):
            try:
                syn = float(pid.get_pi(((0, 1),)))
                red = float(pid.get_pi(((0,), (1,))))
                u0 = float(pid.get_pi(((0,),)))
                u1 = float(pid.get_pi(((1,),)))
            except Exception as e:
                return {"error": f"PID API: {e}", "n_unique": len(outcomes)}

        return {
            "synergy": syn, "redundancy": red,
            "unique_0": u0, "unique_1": u1,
            "n_unique": len(outcomes),
        }
    except Exception as e:
        return {"error": str(e), "n_unique": len(outcomes)}


# ============================================================
# 5. H4 TEST
# ============================================================

def load_entropy_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def test_h4(pid_results: dict, entropy_results: dict, coupling_results: dict) -> dict:
    """
    Test H4 using both PID and raw coupling metrics.

    Primary: PID redundancy slope vs entropy slope (original H4)
    Secondary: delta coupling vs entropy slope (new metric)
    """
    # PID-based H4
    redundancy_slopes = []
    entropy_slopes = []
    session_ids = []

    for sid, traj in pid_results.items():
        if sid not in entropy_results:
            continue
        valid = [p for p in traj if isinstance(p.get("redundancy"), (int, float))
                 and not np.isnan(p.get("redundancy", np.nan))]
        if len(valid) < 3:
            continue

        rounds = [p["round"] for p in valid]
        reds = [p["redundancy"] for p in valid]
        r_slope = stats.linregress(rounds, reds).slope

        H = entropy_results[sid].get("H_trajectory", [])
        if len(H) < 3:
            continue
        h_slope = stats.linregress(range(len(H)), H).slope

        redundancy_slopes.append(r_slope)
        entropy_slopes.append(h_slope)
        session_ids.append(sid)

    # Coupling-based H4 (supplementary)
    coupling_slopes = []
    coupling_entropy_slopes = []
    coupling_sids = []

    for sid, traj in coupling_results.items():
        if sid not in entropy_results:
            continue
        if len(traj) < 3:
            continue

        rounds = [p["round"] for p in traj]
        couplings = [p["cumulative_mean_coupling"] for p in traj]
        c_slope = stats.linregress(rounds, couplings).slope

        H = entropy_results[sid].get("H_trajectory", [])
        if len(H) < 3:
            continue
        h_slope = stats.linregress(range(len(H)), H).slope

        coupling_slopes.append(c_slope)
        coupling_entropy_slopes.append(h_slope)
        coupling_sids.append(sid)

    result = {"pid_h4": {}, "coupling_h4": {}}

    # PID H4
    if len(redundancy_slopes) >= 5:
        r, p = stats.pearsonr(redundancy_slopes, entropy_slopes)
        result["pid_h4"] = {
            "status": "complete",
            "n_sessions": len(redundancy_slopes),
            "pearson_r": round(r, 4),
            "pearson_p": round(p, 6),
            "supported": r < -0.5 and p < 0.05,
        }
    else:
        result["pid_h4"] = {
            "status": "insufficient_data",
            "n_sessions": len(redundancy_slopes),
        }

    # Coupling H4
    if len(coupling_slopes) >= 5:
        r, p = stats.pearsonr(coupling_slopes, coupling_entropy_slopes)
        result["coupling_h4"] = {
            "status": "complete",
            "n_sessions": len(coupling_slopes),
            "pearson_r": round(r, 4),
            "pearson_p": round(p, 6),
            "interpretation": (
                "Coupling increases as entropy increases — coordinated exploration"
                if r > 0.3 and p < 0.1 else
                "Coupling decreases as entropy increases — genuine independence"
                if r < -0.3 and p < 0.1 else
                "No significant coupling-entropy relationship"
            ),
        }
    else:
        result["coupling_h4"] = {
            "status": "insufficient_data",
            "n_sessions": len(coupling_slopes),
        }

    return result


# ============================================================
# 6. VISUALIZATION
# ============================================================

def plot_coupling_trajectory(coupling_traj: list[dict], session_id: str, output_dir: str):
    """Plot delta coupling metrics over rounds."""
    if not HAS_PLT:
        return

    rounds = [c["round"] for c in coupling_traj]
    sim = [c["delta_cosine_simultaneous"] for c in coupling_traj]
    a_leads = [c["delta_cosine_a_leads"] for c in coupling_traj]
    b_leads = [c["delta_cosine_b_leads"] for c in coupling_traj]
    cum = [c["cumulative_mean_coupling"] for c in coupling_traj]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(rounds, sim, 'b-o', label='Simultaneous', linewidth=2, markersize=5)
    ax1.plot(rounds, a_leads, 'g--^', label='A leads B', alpha=0.7, markersize=4)
    ax1.plot(rounds, b_leads, 'm--v', label='B leads A', alpha=0.7, markersize=4)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Delta Cosine Similarity', fontsize=12)
    ax1.set_title(f'Agent Delta Coupling -- {session_id}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)

    ax2.plot(rounds, cum, 'r-s', linewidth=2, markersize=5)
    ax2.fill_between(rounds, cum, 0, alpha=0.2,
                     color='green', where=[c >= 0 for c in cum])
    ax2.fill_between(rounds, cum, 0, alpha=0.2,
                     color='red', where=[c < 0 for c in cum])
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Cumulative Mean Coupling', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"coupling_{session_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pid_trajectory(traj: list[dict], session_id: str, output_dir: str):
    """Plot PID components over rounds."""
    if not HAS_PLT:
        return

    valid = [p for p in traj if isinstance(p.get("synergy"), (int, float))]
    if not valid:
        return

    rounds = [p["round"] for p in valid]
    syn = [p.get("synergy", 0) for p in valid]
    red = [p.get("redundancy", 0) for p in valid]
    u0 = [p.get("unique_0", 0) for p in valid]
    u1 = [p.get("unique_1", 0) for p in valid]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, syn, 'b-o', label='Synergy', linewidth=2)
    ax.plot(rounds, red, 'r-s', label='Redundancy', linewidth=2)
    ax.plot(rounds, u0, 'g--^', label='Unique (A0)', alpha=0.7)
    ax.plot(rounds, u1, 'm--v', label='Unique (A1)', alpha=0.7)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Information (bits)', fontsize=12)
    ax.set_title(f'PID (Delta-MI) -- {session_id}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"pid_v3_{session_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_coupling_summary(all_coupling: dict, output_dir: str):
    """Summary plot: mean coupling by condition."""
    if not HAS_PLT:
        return

    conditions = {}
    for sid, traj in all_coupling.items():
        # Extract condition from session_id if possible
        cond = sid.rsplit("_", 1)[0] if "_" in sid else "unknown"
        if cond not in conditions:
            conditions[cond] = []
        mean_c = np.mean([c["delta_cosine_simultaneous"] for c in traj])
        conditions[cond].append(mean_c)

    if not conditions:
        return

    cond_names = sorted(conditions.keys())
    means = [np.mean(conditions[c]) for c in cond_names]
    stds = [np.std(conditions[c]) for c in cond_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(cond_names))
    ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=15)
    ax.set_ylabel('Mean Delta Cosine Similarity')
    ax.set_title('Agent Coupling by Condition')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, "coupling_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PID v3 -- Delta Mutual Information for Multi-Agent Debates")
    parser.add_argument("--traces_dir", required=True)
    parser.add_argument("--output_dir", default="./pid_results_v3")
    parser.add_argument("--entropy_file", default=None)
    parser.add_argument("--test_h4", action="store_true")
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--n_bins", type=int, default=3,
                        help="Number of quantile bins for delta discretization")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load & embed
    print("\n=== Loading traces ===")
    traces = load_debate_traces(args.traces_dir)
    traces = compute_embeddings_if_needed(traces, args.embedding_model)

    # Raw coupling (always computed — no discretization needed)
    print("\n=== Computing delta coupling (continuous) ===")
    coupling_results = {}
    for trace in traces:
        sid = trace.get("session_id", "unknown")
        coupling = compute_delta_coupling(trace)
        coupling_results[sid] = coupling

        if coupling:
            mean_c = np.mean([c["delta_cosine_simultaneous"] for c in coupling])
            print(f"  {sid}: mean_coupling={mean_c:.4f}")

    # Save coupling
    coupling_file = os.path.join(args.output_dir, "delta_coupling.json")
    with open(coupling_file, 'w') as f:
        json.dump(coupling_results, f, indent=2)
    print(f"\nCoupling saved to {coupling_file}")

    # PID on discretized deltas
    if HAS_DIT:
        print(f"\n=== Computing PID on discretized deltas (n_bins={args.n_bins}) ===")
        pid_results = compute_pid_delta(traces, n_bins=args.n_bins, pool=args.pool)

        pid_file = os.path.join(args.output_dir,
                                "pid_pooled_v3.json" if args.pool else "pid_trajectories_v3.json")

        # Serialize
        serializable = {}
        for key, val in pid_results.items():
            if isinstance(val, list):
                serializable[key] = []
                for entry in val:
                    clean = {}
                    for k, v in entry.items():
                        if isinstance(v, (np.floating, float)):
                            clean[k] = None if np.isnan(v) else round(float(v), 6)
                        elif isinstance(v, np.integer):
                            clean[k] = int(v)
                        else:
                            clean[k] = v
                    serializable[key].append(clean)
            elif isinstance(val, dict):
                serializable[key] = {
                    k: (round(float(v), 6) if isinstance(v, (np.floating, float)) and not np.isnan(v)
                        else int(v) if isinstance(v, np.integer) else v)
                    for k, v in val.items()
                }
            else:
                serializable[key] = val

        with open(pid_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"PID saved to {pid_file}")
    else:
        pid_results = {}
        print("\nSkipping PID (dit not installed)")

    # Plots
    if args.plot:
        print("\n=== Generating plots ===")
        for sid, traj in coupling_results.items():
            if traj:
                plot_coupling_trajectory(traj, sid, args.output_dir)

        if not args.pool:
            for sid, traj in pid_results.items():
                if isinstance(traj, list) and traj:
                    plot_pid_trajectory(traj, sid, args.output_dir)

        plot_coupling_summary(coupling_results, args.output_dir)

    # H4 test
    if args.test_h4:
        if not args.entropy_file:
            print("\nWARNING: --test_h4 requires --entropy_file.")
        else:
            print("\n=== Testing H4 ===")
            entropy = load_entropy_results(args.entropy_file)
            h4 = test_h4(pid_results, entropy, coupling_results)

            h4_file = os.path.join(args.output_dir, "h4_results_v3.json")
            with open(h4_file, 'w') as f:
                json.dump(h4, f, indent=2)

            print(f"\n{'='*60}")
            print(f"  PID H4: {h4['pid_h4']}")
            print(f"  Coupling H4: {h4['coupling_h4']}")
            print(f"{'='*60}")

    print("\n=== PID v3 complete ===")


if __name__ == "__main__":
    main()
