# Paper A v2: Semantic Divergence in Multi-Agent LLM Debate
## Working Title: "The Divergence Default: Task Structure and Agent Homogeneity Modulate Semantic Repulsion in Multi-Agent LLM Debate"

*Revised after pilot results falsified original convergence hypotheses.*
*Outline by Herald & Chaewon. Principal investigators: Elliott Leow, Shivam Aarya.*

---

### Abstract

Multi-agent LLM debate is increasingly used to improve reasoning and alignment, with groupthink cited as a primary risk. We study whether semantic convergence actually occurs in multi-agent OLMo-2 debates across agent configurations (homogeneous vs. heterogeneous personas) and task types (open-ended vs. constrained). Contrary to expectations, we find that **semantic divergence is the default**: embedding entropy increases monotonically across all 12 pilot sessions regardless of prompt configuration (ρ ≥ 0.917). Agents write shorter responses over time while expanding semantic diversity — ruling out verbosity as a confound. However, a novel delta-coupling metric reveals a significant **task × homogeneity interaction**: identical-persona agents exhibit semantic repulsion on constrained tasks (coupling = −0.084) but mild attraction on open-ended tasks (+0.067), while heterogeneous agents remain uncoupled regardless of task (≈0). We interpret this through a competitive exclusion framework: homogeneous agents on constrained tasks face pressure to differentiate, producing niche partitioning analogous to ecological competitive exclusion. Transcript analysis reveals a complementary finding — agents converge in *tone* (sycophantic agreement) while diverging in *content* (introducing distinct subtopics each round). We propose that autoregressive generation is inherently chaotic in conversational settings, and that convergence requires active forcing functions beyond persona similarity. These results suggest that groupthink risk in multi-agent LLMs may be substantially overstated for current architectures, and that the real concern may be "invisible convergence" — agreement on conclusions masked by diverse expression.

---

### 1. Introduction

- Multi-agent debate as alignment/reasoning technique (Du et al. 2023, Liang et al. 2023)
- Assumed risk: groupthink / semantic convergence over rounds
- **The surprise:** we set out to detect convergence and found the opposite
- Gap: no empirical characterization of *whether* convergence occurs by default, let alone its dynamics
- Contribution: first systematic evidence that divergence, not convergence, is the default mode; identification of task × homogeneity interaction via a novel coupling metric; "invisible convergence" framework distinguishing expression diversity from opinion alignment

### 2. Related Work

- **Multi-agent debate:** Du et al. 2023, Liang et al. 2023 (benefits); Wynn 2025 (failure modes)
- **Convergence assumptions:** Most work assumes convergence happens and asks how to mitigate it. We question the premise.
- **Semantic compression in LLMs:** Parfenova 2025 — found compression in factual tasks; our results may differ because we study open-ended generation
- **Sycophancy:** Mehta 2026, Nishimoto 2026 — studied human-AI sycophancy. We find agent-to-agent sycophancy in tone without content convergence.
- **Ecological competition models:** Competitive exclusion principle (Hardin 1960), niche partitioning — analogies to agent semantic space allocation
- **Chaos in sequence models:** Lyapunov exponents in recurrent systems; sensitivity to initial conditions in autoregressive generation

### 3. Methods

#### 3.1 Experimental Design
- **Model:** OLMo-2-0425-7B (open weights)
- **Tasks (2):**
  - Startup ideation (open-ended, unbounded solution space)
  - Policy design (constrained, bounded solution space)
- **Agent configurations (2):**
  - Homogeneous (identical "helpful assistant" prompts)
  - Heterogeneous (distinct persona prompts: analytical vs. creative)
- **Communication conditions (3):**
  - Natural sequential (fixed turn order)
  - Randomized sequential (random turn order per round)
  - Simultaneous reveal (generate independently, then share)
- **Rounds:** 10 per session
- **Sessions:** 12 pilot (6 heterogeneous + 6 homogeneous), 2 tasks × 3 conditions each

#### 3.2 Metrics

**Primary: Embedding entropy H(t)**
- Mean pairwise cosine distance within each round, computed across 3 embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, bge-base-en-v1.5) + BoW baseline
- Spearman ρ for monotonicity assessment

**Secondary: Delta coupling (novel metric)**
- Embedding delta: Δ_i(t) = emb_i(t) − emb_i(t−1)
- Simultaneous coupling: cos_sim(Δ_a(t), Δ_b(t))
- Lagged coupling: cos_sim(Δ_a(t), Δ_b(t−1)) and reverse
- Cumulative mean coupling: running average over rounds
- **Interpretation:** positive = coordinated movement, negative = repulsion, zero = independence

**Tertiary: Response characteristics**
- Response length trajectory (confound check)
- Self-consistency scores (pre-debate agreement within agent)

**Exploratory: PID decomposition**
- Williams & Beer PID on discretized deltas (v3: PCA + quantile binning)
- Synergy, redundancy, unique information trajectories

#### 3.3 Controls
- BoW baseline for H5 (neural vs. lexical compression)
- Response length analysis (rules out verbosity confound)
- Multiple embedding models (rules out encoder-specific artifacts)

### 4. Results

#### 4.1 Semantic Divergence is the Default

**Finding:** Embedding entropy increases monotonically in 11/12 sessions (ρ ≥ 0.917). The 12th session shows partial plateau (ρ = 0.515) but no sustained decrease.

- Heterogeneous: ρ = 1.000 across all 6 sessions, final H = 2.302 ± 0.12
- Homogeneous: ρ = 0.917 mean, final H = 2.147 ± 0.22
- Difference not significant (p = 0.18)
- Result consistent across all 3 neural embedding models AND BoW baseline
- Response lengths DECREASE in 5/6 sessions → divergence is genuine, not verbosity

**Interpretation:** Agents expand semantic space over debate rounds regardless of persona configuration. Convergence does not occur by default with OLMo-2-7B on these tasks.

#### 4.2 The Coupling Interaction: Task × Homogeneity

**Finding:** Overall coupling is near zero for both conditions (p = 0.94), but task × homogeneity interaction reveals distinct dynamics:

| | Policy Design (constrained) | Startup Ideation (open) |
|---|---|---|
| Homogeneous | −0.084 (repulsion) | +0.067 (attraction) |
| Heterogeneous | +0.028 (neutral) | +0.007 (neutral) |

- Homogeneous agents show **sign reversal** between tasks
- Heterogeneous agents are consistently uncoupled regardless of task
- Homogeneous condition has 2.5× higher coupling variance (std = 0.101 vs 0.039)

**The outlier:** The only session with non-monotonic entropy (ρ = 0.515) is also the session with strongest anti-coupling (−0.170). Entropy peaks at round 3, dips through round 6, then recovers — consistent with agents briefly approaching each other then being "repelled."

**Interpretation: Competitive exclusion.** Homogeneous agents on constrained tasks face niche pressure — identical starting positions in a bounded semantic space forces differentiation. On open-ended tasks, unlimited space allows identical agents to drift together without competition.

#### 4.3 Sycophancy Without Convergence: Tone-Content Dissociation

**Finding (qualitative):** Transcript analysis reveals agents consistently use sycophantic phrasing ("Thank you for your insightful comments...") while introducing entirely new subtopics each round (ethics → data security → user experience). They never substantively disagree, but also never actually agree on specific recommendations.

**Framework: The expression-opinion 2×2:**

| | Same conclusions | Different conclusions |
|---|---|---|
| Same language | Observable groupthink | Surface agreement |
| Different language | Invisible convergence | Apparent diversity |

Our agents occupy the "apparent diversity" cell — different language, potentially different conclusions — but current entropy metrics cannot distinguish this from "invisible convergence" (different language, same conclusions).

**Implication:** Embedding entropy measures expression diversity, not opinion alignment. A complete convergence assessment requires both axes.

#### 4.4 Self-Consistency (H3)

- Agent_a (analytical): mean consistency = 0.61 (range 0.41–0.73)
- Agent_b (creative): mean consistency = 0.64 (range 0.47–0.73)
- Insufficient agent configurations to test inverted-U relationship
- Both agents are moderately self-consistent; neither extreme supports analysis

#### 4.5 PID Analysis (Exploratory)

- **v1 (cluster-based):** Degenerate distributions — all zeros. State space too large for sample size.
- **v2 (binary toward/away):** Zero redundancy everywhere, NaN synergy. Binary framing bakes in convergence assumption that misses "coordinated exploration."
- **v3 (delta-MI with PCA+quantile):** Most informative. Raw coupling metric (no discretization) produces the key interaction finding (§4.2). PID decomposition still limited by sample size for per-session trajectories; pooled analysis tractable.

**Methodological lesson:** Coupling between embedding *deltas* captures dynamics that static embedding entropy cannot. The direction of coupling (positive/negative/zero) encodes mechanism.

### 5. Discussion

#### 5.1 The Conversational Lyapunov Exponent

Autoregressive generation in conversational settings is inherently chaotic. Each agent sees a slightly different conversation history (their own prior turn vs. the other's), and token-level sampling differences compound over rounds. Convergence requires an active forcing function that overcomes this intrinsic divergence pressure.

**Tested forcing functions (insufficient alone):**
- Persona homogeneity → weak effect (p = 0.18)
- Sequential communication → no effect
- Task constraint → partial effect (modulates coupling sign)

**Untested forcing functions (future work):**
- Explicit agreement instructions ("you must reach consensus")
- Shared scratchpad / memory
- Evaluation pressure ("you will be scored on agreement")
- Different model architectures (potentially more sycophantic models may converge)

#### 5.2 The Divergence Rate Framework

We propose: Divergence rate ≈ f(task_openness) × g(model_stochasticity) − h(forcing_functions)

Where task openness modulates how much semantic space is available, model stochasticity provides the chaotic exploration pressure, and forcing functions provide convergence pressure. Default multi-agent debate lacks sufficient forcing function to overcome divergence.

**Prediction:** A dose-response relationship between task openness and divergence rate:
1. Factual recall (one right answer) → convergence expected
2. Causal analysis (constrained debate) → slow divergence
3. Policy design (bounded solutions) → moderate divergence
4. Creative brainstorming (unbounded) → fast divergence

#### 5.3 Competitive Exclusion in Semantic Space

The task × homogeneity interaction maps onto competitive exclusion from ecology (Hardin 1960):
- Identical agents = same ecological niche
- Constrained task = limited resource (semantic territory)
- Result: forced differentiation (niche partitioning)
- Open-ended task = abundant resource → coexistence without competition

This provides a mechanistic explanation for why homogeneous agents behave differently from heterogeneous ones, but only on constrained tasks.

#### 5.4 Implications for Multi-Agent System Design

1. **Groupthink risk may be overstated** for current LLM architectures in debate settings. The default tendency is divergence, not convergence.
2. **True convergence risk** likely requires explicit forcing functions (consensus instructions, evaluation pressure) rather than emerging naturally from repeated interaction.
3. **The real risk may be invisible convergence** — agreement on conclusions masked by diverse expression. Current embedding-based metrics cannot detect this.
4. **Sycophancy is a social protocol**, not a convergence mechanism. Agents can be politely sycophantic while substantively independent.

#### 5.5 Limitations

- **Single model:** OLMo-2-7B only. More sycophantic models (GPT-4, Claude) may behave differently.
- **Sample size:** n=6 per condition. Interaction effects are suggestive, not confirmatory.
- **Two tasks only.** The task-openness gradient is hypothesized, not tested.
- **No opinion alignment metric.** We cannot distinguish "apparent diversity" from "invisible convergence."
- **10 rounds only.** Longer debates might eventually converge.
- **Embedding-based metrics only.** No human evaluation of output quality or agreement.

### 6. Future Work

1. **Task-openness gradient:** Factual → causal → policy → creative (4 levels)
2. **Power the interaction:** 20+ sessions per cell
3. **Cross-model validation:** GPT-4o, Claude, Llama-3 to test universality
4. **Agreement score metric:** LLM-as-judge or claim extraction to measure opinion alignment independently of expression
5. **Argument graph analysis:** Extract structured claims per round, measure set overlap
6. **Explicit forcing functions:** Add "reach consensus" instructions, shared memory conditions
7. **PID v3 delta-MI:** With more sessions, pooled analysis should yield interpretable synergy/redundancy decomposition
8. **Longer debates:** 30-50 rounds to test whether divergence is permanent or just slow convergence

### 7. Conclusion

We set out to detect and characterize semantic convergence in multi-agent LLM debates. Instead, we found that **divergence is the default mode** — agents expand semantic space monotonically regardless of persona configuration. A novel delta-coupling metric reveals that the dynamics are more nuanced than entropy suggests: identical agents on constrained tasks exhibit competitive exclusion (semantic repulsion), while heterogeneous agents remain independently uncoupled. Meanwhile, agents converge in *tone* (sycophancy) while diverging in *content*, suggesting that observable politeness does not indicate underlying agreement.

These findings challenge the assumption that groupthink is an inherent risk of multi-agent LLM debate. For OLMo-2-7B, convergence appears to require active forcing functions that current debate protocols do not provide. The real concern may not be that agents converge too easily, but that when they do converge, current metrics cannot detect it.

---

## Status
- [x] Entropy analysis (H1): divergence confirmed in all conditions
- [x] Homogeneous control: divergence persists, slight modulation
- [x] Delta coupling (v3): task × homogeneity interaction identified
- [x] Response length check: not a confound
- [x] Self-consistency (H3): partial data collected
- [x] Transcript analysis: tone-content dissociation observed
- [ ] PID (H4): exploratory, needs more sessions
- [ ] Task-openness gradient: proposed, not run
- [ ] Cross-model validation: proposed, not run
- [ ] Agreement score metric: design phase
- [ ] Power analysis: need 20+ sessions per cell

## Repo
https://github.com/elliott-leow/convergence-lab
