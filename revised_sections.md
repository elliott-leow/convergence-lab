# Revised Sections for Paper v5

## §4.3 Forcing Functions: Divergent Effects

Table 1 summarizes the forcing function results, compared against matched baselines (the same task × persona combinations from Experiment 1).

[TABLE 1 — updated with matched baseline column]

The most striking finding is that consensus instruction and evaluation pressure have opposite effects on endpoint similarity. Evaluation pressure improves endpoint convergence relative to matched baselines (+0.070 for heterogeneous, +0.132 for homogeneous), while consensus instruction paradoxically *decreases* it (−0.162 for heterogeneous, −0.021 for homogeneous).

Both conditions reduce ρ relative to baseline, indicating slower divergence trends. But slower divergence does not guarantee convergent endpoints. Consensus-instructed agents diverge more slowly in trajectory while ending up further apart—a pattern consistent with agents casting wider semantic nets in search of "common ground," introducing more diverse content in the process.

Evaluation pressure, by contrast, achieves high endpoint similarity (0.926 for homogeneous agents) despite a still-positive ρ. We hypothesize that the critical distinction is specificity: the consensus condition provides a vague behavioral target ("reach agreement") that expands the content search space, while the evaluation condition provides a specific one ("an expert judge will score engagement with your partner's points") that constrains it.

This distinction—between instructions that sound convergent and instructions that actually produce convergence—has practical implications for multi-agent system design. Not all convergence mandates converge. The consensus paradox suggests that naive instructions to agree may be a divergent instruction disguised as a convergent one.


## §5.2 A Hierarchy of Convergence Forces (revised)

Our experiments reveal a hierarchy of forces acting on convergence, with a surprising inversion at the instruction level:

1. **Temperature reduction** (most effective). Greedy decoding eliminates divergence entirely by removing the stochastic perturbations that drive it. However, it produces degenerate output (identical text), making it impractical alone.

2. **Evaluation pressure** (partially effective). Framing the debate as a scored evaluation achieves high endpoint similarity (0.926 for homogeneous agents) by incentivizing engagement with the partner's specific arguments. This constrains the content space, producing convergent endpoints within the sampling noise budget.

3. **No instruction** (baseline). Standard debate without convergence instructions produces natural endpoint similarity of ~0.80—higher than previously estimated, indicating that agents sharing a conversational context naturally develop moderate similarity even while diverging in trend.

4. **Consensus instruction** (counterproductive). Explicitly telling agents to reach consensus produces lower endpoint similarity than saying nothing (−0.162 for heterogeneous agents). The instruction appears to expand rather than constrain the semantic search space, as agents explore broader territory in pursuit of vaguely defined "common ground."

The practical implication is nuanced: multi-agent systems requiring convergence should use evaluation-aware prompting with specific behavioral targets, not generic consensus mandates. The difference between "agree" and "engage with your partner's specific points" is the difference between divergence and convergence.

This finding also connects to our tone-content dissociation observation (§4.5). Consensus-instructed agents may increase sycophantic framing (trying harder to *sound* agreeable) while the instruction's vagueness drives their actual content further apart. Sycophancy and convergence are not only independent—under consensus pressure, they may be inversely related.
