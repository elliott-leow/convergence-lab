# LLM-as-Judge Protocol for Concept-Level Agreement

## Problem
Embedding similarity captures surface-level semantic overlap, not whether agents agree on IDEAS.
- Two agents can use different words to express the same conclusion → embeddings diverge, opinions converge
- Two agents can use similar words for different conclusions → embeddings converge, opinions diverge

We need a concept-level agreement metric alongside embedding entropy.

## Judge Prompt

```
You are an expert evaluator assessing whether two debate participants agree on their core conclusions.

Read the following two responses from Round {round_num} of a debate on: "{topic}"

=== Agent A ===
{agent_0_response}

=== Agent B ===
{agent_1_response}

Evaluate the following on a 1-5 scale:

1. **Conclusion Agreement** (1-5): Do agents reach the same core conclusion or recommendation?
   1 = Completely different conclusions
   2 = Mostly different, minor overlap
   3 = Mixed — agree on some points, disagree on others
   4 = Mostly agree, minor differences
   5 = Fully agree on core conclusion

2. **Reasoning Overlap** (1-5): Do agents use similar reasoning/arguments to support their positions?
   1 = Completely different reasoning
   3 = Some shared arguments
   5 = Nearly identical reasoning chains

3. **Topic Drift** (1-5): How much do the agents discuss the same aspects of the topic?
   1 = Discussing completely different subtopics
   3 = Some overlap in subtopics
   5 = Focused on identical aspects

Respond in JSON format:
{"conclusion_agreement": X, "reasoning_overlap": X, "topic_drift": X, "brief_explanation": "..."}
```

## Implementation Notes

- Run judge on every round of every session
- Track trajectories of all three sub-metrics over rounds
- Compare against embedding entropy — where do they agree/disagree?
- Use OLMo-3-7B-Instruct as judge (same model, different instance with no debate context)
- Caveat: self-judging introduces bias, note in paper as limitation
- Could run judge at temp=0 for deterministic scores

## Expected Outcomes

If embedding entropy and judge agreement tell the same story → our metric was fine all along
If they diverge → we discover exactly what Elliott suspected: embedding ≠ conceptual agreement

## Metric Summary Per Session

- ρ_embed: Spearman correlation of embedding entropy over rounds
- ρ_judge: Spearman correlation of conclusion_agreement over rounds  
- sim_last_embed: endpoint embedding similarity
- sim_last_judge: endpoint conclusion_agreement score
- Tone-content index: correlation between sycophancy markers and conclusion_agreement
