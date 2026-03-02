# Paper A: Convergence Failure Detection Experiments

## Hypotheses
- H1: Embedding entropy H(t) decreases monotonically over debate rounds
- H2: Per-agent δ(t) declines before group H(t) collapses (leading indicator)
- H3: Pre-debate self-consistency has inverted-U relationship with debate quality
- H4: PID redundancy increase correlates with entropy decrease
- H5: Compression trajectory is consistent across neural embedders but absent in BoW

## Experiment Structure
1. `run_debates.py` — Run multi-agent debate sessions using OLMo 2 locally
2. `measure_entropy.py` — Compute embedding entropy H(t) and per-agent δ(t)
3. `self_consistency.py` — Pre-debate solo consistency measurement
4. `analyze.py` — Statistical analysis and hypothesis testing
5. `pid_analysis.py` — Partial Information Decomposition

## Task Battery (5 types × 2 instantiations × 3 conditions = 30 sessions)
1. Startup ideation
2. Policy design
3. Research question generation
4. Creative problem-solving
5. Ethical dilemma

## Communication Conditions
- A: Natural sequential (agent 1 → agent 2 → agent 1...)
- B: Randomized sequential (random order each round)
- C: Simultaneous reveal (both generate, then both see)
