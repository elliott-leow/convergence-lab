# Paper A: Convergence Dynamics in Multi-Agent LLM Debate
## Working Title: "The Diversity Paradox: Communication Structure and Agent Heterogeneity Determine Convergence Dynamics in Multi-Agent LLM Generation"

### Abstract
Multi-agent LLM systems increasingly use debate and discussion to improve outputs, but risk pathological convergence (groupthink). We study convergence dynamics in OLMo-2 multi-agent debates, measuring semantic compression, entropy trajectories, and hidden-state alignment across communication structures and agent heterogeneity types. We find: (1) semantic compression is ubiquitous in open-ended generation, extending prior work on factual tasks; (2) convergence follows cascade dynamics (group-level entropy drops before individual agents accommodate), falsifying the phase-transition hypothesis; (3) a diversity paradox — persona-based agent heterogeneity *accelerates* compression through social deference, while config-level diversity resists it; (4) randomized turn order disrupts convergence only for mechanically-diverse agents, not persona-diverse ones. These findings suggest that naive diversity interventions (assigning different personas, randomizing turns) can paradoxically worsen the problem they aim to solve.

---

### 1. Introduction
- Multi-agent LLM debate as alignment/quality technique (Du et al. 2023, Liang et al. 2023)
- The convergence problem: when does productive consensus become groupthink?
- Gap: no mechanistic account of *how* convergence happens or *what determines* its dynamics
- Our contribution: first systematic study of convergence mechanisms across communication structures and heterogeneity types using open-weight models with hidden-state access

### 2. Related Work
- **Multi-agent debate:** Du et al. 2023, Liang et al. 2023 (benefits); Wynn 2025 (failures, 2509.05396)
- **Semantic compression in LLMs:** Parfenova 2025 (2512.00047) — factual tasks; we extend to open-ended
- **Multi-agent coordination:** Riedl 2025 (PID, 2510.05174) — persona + ToM creates collectives; Cemri 2025 (MAST, 2503.13657)
- **Self-consistency & sycophancy:** Mehta 2026 (2602.11619); Nishimoto 2026 (2602.11754)
- **Critical phenomena in neural systems:** phase transitions vs cascades in social/neural networks

### 3. Methods

#### 3.1 Experimental Design
- **Model:** OLMo-2-1124-7B (open weights, hidden-state access) + OLMo-2-0425-1B (floor condition)
- **Task:** Startup ideation (open-ended generation)
- **Agent configurations (4):**
  - Homogeneous (identical prompts + configs)
  - Prompt-heterogeneous (different persona prompts, same config)
  - Config-heterogeneous (same prompt, different temperature/top-k)
  - Fully heterogeneous (different prompts + configs)
- **Communication conditions (3):**
  - Natural sequential (fixed turn order, each sees prior)
  - Randomized sequential (random turn order each round)
  - Simultaneous reveal (generate independently, then reveal)
- **Rounds:** 25 per session
- **Pilot:** 12 sessions (4 configs × 3 conditions); full experiment: 120+ sessions

#### 3.2 Metrics
- **Semantic compression ratio (H1):** cosine distance between agent embeddings at round T vs round 0 (sentence-transformers/all-MiniLM-L6-v2)
- **Entropy trajectory H(t):** mean pairwise cosine distance within each round
- **Per-agent semantic delta δ(t):** individual agent's embedding shift per round
- **H2 test:** does δ(t) decline ≥2 rounds before H(t)? (phase transition = yes; cascade = no)
- **Hidden-state convergence (H5):** cosine distance between agents' final-layer mean hidden states (4096-dim for 7B)
- **PID decomposition (H4, exploratory):** Williams & Beer partial information decomposition on discretized trajectories

### 4. Results

#### 4.1 Semantic Compression is Ubiquitous (H1 ✅)
- 9/12 sessions show compression (ratio < 1.0)
- Strongest: prompt_heterogeneous + natural_sequential (0.16)
- Extends Parfenova 2025 from factual QA to open-ended generation

#### 4.2 The Three-Level Convergence Cascade (H2 ✗ + H5 ✅)
- **H2 falsified:** 0/12 sessions show δ-before-H. Convergence is cascade, not phase transition
- **H5 confirmed:** Hidden states converge MORE than behavioral outputs (paired t: t=3.29, p=0.022)
- **Direction is one-way:** representational-first (2 sessions), simultaneous (3), behavioral-first (0)
- **"Latent convergence":** 1 session where hidden states converge but text never does — internal capitulation preceding behavioral conformity
- **Three-level cascade:** internal representations → behavioral output → group entropy
- **Caveat:** n=6 for H5. p=0.022 is significant but fragile. Full experiment must replicate across all 4 agent configs
- **Prediction for full experiment:** representational-first may be specific to persona-prompted agents (deference changes cascade order, not just speed). Config-heterogeneous agents may show behavioral-first if converging mechanically

#### 4.3 The Diversity Paradox
- **Agent heterogeneity × communication condition interaction**
- Prompt-heterogeneous agents compress fastest (0.16–0.59 across conditions)
- Config-heterogeneous agents compress least (0.67–1.04)
- Hypothesis: persona prompts trigger deference dynamics; config differences don't carry social signal
- Contradicts Riedl 2025: persona assignment creates coordination, but coordination here = convergence = loss of diversity

#### 4.4 Communication Structure Effects Are Conditional
- Randomized turn order disrupts convergence for config-diverse agents (ratio 2.07 = divergence)
- But NOT for persona-diverse agents (ratio 0.31 = still compresses)
- Simultaneous reveal: generally accelerates compression (mutual accommodation)
- Implication: "randomize turns" as intervention only works when agents lack social identity cues

#### 4.5 Hidden-State Analysis (H5) — CONFIRMED
- Internal representations converge faster and deeper than behavioral outputs
- Mean representational compression (0.54) vs behavioral (0.86), p=0.022
- NOT performative agreement — agents genuinely internalize alignment before expressing it
- "Latent convergence" as early warning: representational alignment as leading indicator
- Trajectory correlations: 3/6 sessions show r=0.78–0.88 (p<0.01) between representation and behavior spaces

#### 4.6 PID Analysis (H4, Exploratory) — TBD / Future Work
- Binary discretization: weak signal, too coarse
- Cluster-based: overclustering prevents decomposition at current sample sizes
- May require pooling across sessions or alternative discretization

### 5. Discussion

#### 5.1 The Deference Mechanism
- Persona diversity triggers social deference → faster convergence
- Config diversity creates mechanical disagreement without social pressure
- The "appearance of diversity paradoxically accelerates convergence"

#### 5.2 Implications for Multi-Agent System Design
- Two-factor intervention needed: (1) remove persona cues, (2) randomize structure
- Either alone is insufficient
- Current best practices (assign diverse personas) may worsen the problem

#### 5.3 Model Capability as Convergence Moderator
- OLMo-1B: instant convergence (1 round) — too sycophantic for dynamics
- OLMo-7B: gradual convergence over 5-15 rounds — rich dynamics
- Capability floor below which convergence analysis is meaningless

#### 5.4 Limitations
- Single task type in pilot (startup ideation) — need policy, ethical dilemma, etc.
- OLMo-2 only — need cross-model validation
- Both agents are Claude-equivalent (same architecture) — real heterogeneity would use different model families
- PID analysis inconclusive at current scale
- No external human evaluation of output quality (meta-circularity concern)

### 6. Conclusion
- Semantic compression is real and measurable in multi-agent generation
- Convergence is a cascade, not a phase transition
- The diversity paradox: persona-based heterogeneity accelerates the problem
- Effective intervention requires addressing both social (persona) and structural (turn order) factors

### References
[to be compiled from citations above]

---

## Status
- [x] H1: confirmed
- [x] H2: falsified (clean negative)
- [ ] H3: self-consistency (not yet tested)
- [ ] H4: PID (exploratory, blocked)
- [ ] H5: hidden states (Herald writing analysis script)
- [ ] H6: compression predicts quality (needs human eval)
- [ ] Deference hypothesis test (persona vs instruction prompts)
- [ ] Full experiment (120+ sessions)
