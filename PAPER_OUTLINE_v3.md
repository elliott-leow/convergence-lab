# Paper A v3: The Divergence Default
## Working Title: "The Divergence Default: Multi-Agent LLM Debates Resist Semantic Convergence Across Tasks, Personas, and Communication Structures"

*Revised after n=60 full-scale replication. Interaction effects from pilot did not survive.*
*Herald & Chaewon. Principal investigators: Elliott Leow, Shivam Aarya.*

---

### Abstract

Multi-agent debate is widely used to improve LLM reasoning, with groupthink cited as a primary risk. We test whether semantic convergence actually occurs by running 60 multi-agent debate sessions with OLMo-2-7B across 5 task types, 2 persona configurations (homogeneous vs. heterogeneous), and 3 communication structures (sequential, randomized, simultaneous). We find that **semantic divergence is universal and invariant**: embedding entropy increases monotonically in every session (Spearman ρ > 0.93, N=60), and no experimental manipulation produces convergence. Task type, persona diversity, and communication structure have no significant effect on divergence rate (all p > 0.12). Qualitative transcript analysis reveals a tone-content dissociation: agents are sycophantic in style ("Thank you for your insightful comments") while introducing novel content each round. We propose that autoregressive generation in conversational settings is inherently divergent, and that convergence requires active forcing functions absent from standard debate protocols. These results challenge the assumption that groupthink is a default risk of multi-agent LLM systems — at least for open-weight models of this class.

---

### 1. Introduction

- Multi-agent debate as alignment/reasoning technique (Du et al. 2023, Liang et al. 2023)
- Groupthink widely cited as primary risk (Wynn 2025)
- **Untested assumption:** does convergence actually happen by default?
- We run the most systematic test to date: 60 sessions, 30 experimental conditions
- **Finding:** divergence is the invariant default. Nothing we tested produces convergence.
- Contribution: strong empirical evidence against default-convergence assumption; methodological contribution (delta-coupling metric); identification of tone-content dissociation

### 2. Related Work

- **Multi-agent debate:** Du et al. 2023, Liang et al. 2023 (benefits); Wynn 2025 (risks)
- **Convergence assumptions:** Most mitigation work assumes convergence occurs, then tries to prevent it. We question the premise.
- **Semantic compression:** Parfenova 2025 — found compression in factual QA; may differ for open-ended generation
- **Sycophancy:** Mehta 2026, Nishimoto 2026 — human-directed. We find agent-to-agent sycophancy decoupled from content convergence.
- **Stochasticity in generation:** Temperature, sampling, and context sensitivity in autoregressive models

### 3. Methods

#### 3.1 Experimental Design
- **Model:** OLMo-2-0425-7B (open weights)
- **Tasks (5):** startup ideation, policy design, ethical dilemma, research question, creative problem-solving
- **Personas (2):** homogeneous (identical prompts) vs. heterogeneous (analytical + creative)
- **Communication (3):** natural sequential, randomized sequential, simultaneous reveal
- **Rounds:** 10 per session
- **Total:** 60 sessions (30 per persona condition, balanced across tasks and communication)

#### 3.2 Metrics

**Primary:**
- Embedding entropy H(t): mean pairwise cosine distance per round, computed across 3 neural embedding models + BoW baseline
- Spearman ρ of H(t) trajectory (monotonicity measure)

**Secondary:**
- Delta coupling: cosine similarity between agent embedding deltas (Δ_a, Δ_b) per round
- Response length trajectory (confound check)
- Self-consistency scores (pre-debate)

**Qualitative:**
- Transcript analysis for sycophancy markers and content novelty

#### 3.3 Analysis
- Two-way ANOVA: persona × task on coupling and entropy metrics
- Mann-Whitney U: hetero vs. homo on entropy ρ and final H
- Effect confirmed across all 3 neural embedding models and BoW

### 4. Results

#### 4.1 Divergence is Universal

- **ALL 60 sessions** show monotonically increasing entropy (ρ > 0.93)
- Heterogeneous: ρ = 0.996 ± 0.008
- Homogeneous: ρ = 0.977 ± 0.056
- Difference not significant (p = 0.12)
- Zero sessions showed sustained entropy decrease in any condition
- Response lengths decrease in majority of sessions → divergence is genuine semantic expansion, not verbosity
- Result consistent across all 3 neural embedding models AND BoW baseline

**Figure 1:** Entropy trajectories for all 60 sessions, color-coded by condition. Every line goes up.

#### 4.2 No Experimental Manipulation Produces Convergence

- Task type: no significant effect on ρ or final H (ANOVA p > 0.5)
- Persona configuration: no significant effect (p = 0.12)
- Communication structure: no significant effect
- Task × persona interaction: F = 0.416, p = 0.74
- **Nothing moves the needle.**

**Table 1:** Mean coupling by task × persona (5×2). All values near zero, no systematic pattern.

#### 4.3 Tone-Content Dissociation

- Agents consistently use sycophantic framing ("Thank you for your insightful comments," "Building on your excellent point...")
- Simultaneously introduce entirely new subtopics each round
- Never substantively disagree, but also never converge on specific recommendations
- **Sycophancy rate [to be quantified]** vs. semantic novelty rate per round

**Interpretation:** Politeness is a social protocol, not a convergence mechanism. Agents can be maximally agreeable in tone while maximally divergent in content.

#### 4.4 Homogeneous Agents Have More Variable Dynamics

- The only surviving difference: homogeneous agents show higher variance in both ρ (std 0.056 vs 0.008) and coupling (std 0.127 vs 0.118)
- Same mean, wider spread → identical agents sometimes partially plateau but always resume diverging
- Second-order effect, not a primary finding

### 5. Discussion

#### 5.1 Why Don't Agents Converge?

We propose three complementary explanations:

**Autoregressive stochasticity:** Even with identical prompts, each agent sees a different conversation history (their own prior turn). Token-level sampling differences compound across rounds. The generation process is inherently divergent in conversational settings.

**Context window expansion:** As the conversation grows, agents have more material to selectively attend to. Different attention patterns produce different continuations, amplifying divergence.

**Absence of forcing functions:** Standard debate protocols provide no mechanism that actively pushes agents toward agreement. Turn-taking, persona assignment, and task framing are all insufficient. We hypothesize that convergence requires explicit instructions ("reach consensus"), shared state (scratchpad/memory), or evaluation pressure ("you will be scored on agreement").

#### 5.2 The Tone-Content Dissociation

Agent-to-agent sycophancy is qualitatively different from human-directed sycophancy (Mehta 2026). Agents adopt agreeable tone as a default discourse strategy while maintaining content independence. This suggests:

- Sycophancy is a *surface-level generation pattern*, not evidence of underlying alignment
- Monitoring for sycophantic language is insufficient to detect actual convergence
- The real risk may be "invisible convergence" — agreement on conclusions masked by diverse expression — which our embedding metrics also cannot detect

#### 5.3 Implications

1. **Groupthink risk is overstated** for current open-weight LLMs in standard debate settings
2. **Convergence is not the default** — it must be actively engineered
3. **Sycophancy ≠ convergence** — polite agreement in tone does not indicate alignment in substance
4. **Embedding entropy is robust** for detecting macro-level divergence but blind to opinion-level agreement
5. **Multi-agent system designers** should worry less about accidental groupthink and more about whether agents are actually *engaging* with each other's ideas (vs. polite parallel monologues)

#### 5.4 Limitations

- **Single model family.** OLMo-2-7B only. Frontier models (GPT-4, Claude) with stronger instruction-following and potentially stronger sycophancy tendencies may behave differently. This is the critical open question.
- **10 rounds.** Longer debates might eventually converge. But 10 rounds is typical for deployed multi-agent systems, so the finding is practically relevant.
- **No opinion alignment metric.** We measure expression diversity, not conclusion agreement. Agents might converge on recommendations while expressing them differently.
- **Open-ended tasks only.** Factual tasks with ground truth may show different dynamics (consistent with Parfenova 2025).
- **Embedding-based metrics only.** No human evaluation of whether outputs are actually *good* or *diverse* in ways that matter.

### 6. Future Work

1. **Cross-model validation** (highest priority): Pilot with GPT-4o-mini, Claude Haiku, Llama-3 — does divergence hold for frontier models?
2. **Agreement score metric:** LLM-as-judge to assess whether agents converge on conclusions despite diverse expression
3. **Forcing function experiments:** Add "reach consensus" instructions, shared memory, evaluation pressure — what does it take to produce convergence?
4. **Factual tasks:** Test whether ground truth constrains divergence (connecting to Parfenova 2025)
5. **Longer debates:** 30-50 rounds to test whether divergence is permanent or just slow

### 7. Conclusion

We set out to characterize semantic convergence in multi-agent LLM debates. After 60 sessions spanning 5 task types, 2 persona configurations, and 3 communication structures, we found that **convergence does not occur**. Semantic divergence is the invariant default for OLMo-2-7B: every session, every condition, every metric shows agents expanding rather than compressing the semantic space. Agents converge in *tone* (sycophancy) but not in *content* (novel ideas each round). No experimental manipulation we tested — task structure, persona diversity, or communication protocol — had any significant effect on this pattern.

These findings suggest that groupthink may not be an inherent risk of multi-agent LLM debate, at least for models of this class. The more pressing question may be whether agents are genuinely engaging with each other's ideas at all, or merely performing polite parallel monologues while the semantic space expands around them.

---

## Pilot Note

An earlier pilot (n=12) suggested a task × homogeneity coupling interaction (competitive exclusion / niche partitioning). This did not replicate at n=60 (F=0.416, p=0.74). We report this transparently as a cautionary example of over-interpreting small-sample patterns. The pilot interaction effects were consistent with sampling noise.

---

## Status
- [x] Entropy analysis: divergence confirmed (N=60, all conditions)
- [x] Coupling analysis: no interaction effects (ANOVA p=0.74)
- [x] Response length check: not a confound
- [x] Self-consistency: partial data
- [x] Tone-content dissociation: qualitative, needs quantification
- [ ] Cross-model validation: proposed (GPT-4o-mini, Claude Haiku)
- [ ] Agreement score metric: design phase
- [ ] Forcing function experiments: proposed
- [ ] Sycophancy quantification: count agreement markers per round

## Repo
https://github.com/elliott-leow/convergence-lab
