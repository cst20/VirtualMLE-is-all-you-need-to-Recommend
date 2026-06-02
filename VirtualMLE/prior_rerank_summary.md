# SASRec Prior and Rerank Summary

This note summarizes how `prior` and `rerank` are implemented in the main experimental SASRec runner and checks whether the current `VirtualMLE` markdown files already contain hints about possible structural improvements.

## 1. What `prior` means in `SASRec/run_sasrec.py`

In the main runner, a `prior` is an extra score derived from train-split statistics and added to the model logits at evaluation time. It is not a separate neural module; it is a lightweight statistical signal computed from the training split only.

### Supported prior types

1. **Popularity prior**
   - Controlled by `--popularity_bias_alpha`.
   - Adds a normalized log-popularity bias to item scores.
   - Useful when the domain is sparse and popularity is a strong signal.

2. **Transition prior**
   - Controlled by `--transition_prior_alpha`.
   - Uses adjacent item transitions from the training split.
   - Intuition: if item `A` is often followed by `B`, boost `B` when the latest history item is `A`.

3. **ItemCF prior**
   - Controlled by `--itemcf_prior_alpha`.
   - Uses train-split item co-occurrence inside a configurable local window.
   - Intuition: if items often co-occur in short histories, use that co-occurrence to refine ranking.

### History prior controls

- `--history_prior_topk`: keep only top-K neighbors per source item.
- `--history_prior_recent_k`: use only the most recent K history items when applying priors.
- `--history_prior_norm`: normalize multi-history itemCF contribution by `sqrt(count)`, `count`, or `none`.
- `--itemcf_window_size`: control the co-occurrence window when building the itemCF prior.
- `--score_normalization`: optionally normalize model logits row-wise before prior blending.
- `--dense_history_prior`: materialize dense GPU prior matrices for faster evaluation-time blending.

### Effective scoring view

The evaluation-time score can be viewed as:

```text
final_score
= normalize(model_score)
+ transition_alpha * transition_prior
+ itemcf_alpha * itemcf_prior
+ popularity_alpha * popularity_prior
```

This happens only after the base SASRec model has produced full-item logits.

## 2. What `rerank` means in `SASRec/run_sasrec.py`

`rerank` is a two-stage refinement strategy.

Instead of adding prior signals to all items, the script first selects the current top-K items from the model scores, then applies extra prior bonuses only inside that candidate set.

### Supported rerank controls

- `--rerank_topk`
- `--rerank_transition_alpha`
- `--rerank_itemcf_alpha`
- `--rerank_popularity_alpha`

### Effective rerank flow

```text
1. Compute full model scores.
2. Select top-K candidates.
3. Build rerank bonuses from transition / itemCF / popularity priors.
4. Apply the bonus only to the selected candidate set.
5. Re-sort the candidates and produce final top-K results.
```

### Difference from direct prior blending

- **Prior blending** affects the full item space.
- **Rerank** affects only the current top-K candidates.

In practice:

- Prior blending is stronger and can change global recall behavior.
- Rerank is more conservative and often safer when the base model is already a decent retriever.

## 3. Why both are useful

### Prior blending is useful when:

- the model is still weak,
- the domain is sparse,
- train-split co-occurrence is strong,
- and explicit statistical knowledge should directly influence retrieval.

### Rerank is useful when:

- the base model already recalls reasonable candidates,
- we mainly want to improve top-K ordering,
- and we want to reduce the risk of disturbing the whole score distribution.

## 4. Main experimental runner vs `VirtualMLE` release runner

### Main experimental runner

File:

- `SASRec/run_sasrec.py`

Characteristics:

- contains prior blending,
- contains candidate reranking,
- contains multiple structural search hooks,
- is designed as an experiment-oriented runner.

### `VirtualMLE` release runner

File:

- `VirtualMLE/SASRec/run_sasrec.py`

Characteristics:

- keeps only the minimal SASRec baseline,
- does not contain prior blending,
- does not contain rerank logic,
- does not expose the richer structural switches from the experiment-oriented runner.

## 5. Does `VirtualMLE` markdown already mention possible structural improvements?

### Short answer

**Yes, but only partially.**

The current `VirtualMLE` markdown files do contain some improvement hints, but most of them are still high-level release-backlog suggestions rather than a full detailed design list.

### Generic structural / experimental backlog already present

In the current `VirtualMLE` program reflections, the release cells already mention several search directions:

#### SASRec release cells

- compare `hidden_dim`, `num_blocks`, and `num_heads`,
- compare `Adam` vs `AdamW`,
- compare `all_positions` vs `last_position`,
- test `train_window_size`.

These appear, for example, in:

- `VirtualMLE/SASRec/Baby/program_reflection.md`
- `VirtualMLE/SASRec/Beauty/program_reflection.md`

#### HSTU release cells

- compare `hidden_dim`, `linear_dim`, `attention_dim`, `num_blocks`, `num_heads`,
- compare `AdamW` vs `Adam`,
- compare `all_positions` vs `last_position`,
- test `dropout`, `attn_dropout`, and history truncation.

These appear, for example, in:

- `VirtualMLE/HSTU/Baby/program_reflection.md`
- `VirtualMLE/HSTU/Beauty/program_reflection.md`

### More detailed Baby-specific hints already present

The `Baby` release markdown files already contain more specific transferable hints than the other domains.

#### `VirtualMLE/SASRec/Baby/program_reflection.md`

Already mentions:

- compact tied backbones,
- train-only prior hybrids,
- moderate transition / itemCF prior strength,
- preserving recency with `last` pooling,
- `SWIGLU` being more promising than `relative_bias`,
- widening before deepening,
- continuing prior-oriented search in short-sequence domains.

#### `VirtualMLE/HSTU/Baby/program_reflection.md`

Already mentions:

- strong value of shared embeddings,
- possible benefit from disabling embedding L2 normalization,
- importance of multi-position supervision,
- conservative attention-dropout search,
- train-split co-occurrence priors,
- checking recency dilution when pooling or reranking underperforms.

## 6. What is still missing in `VirtualMLE` markdown

Compared with the richer experimental SASRec runner, the `VirtualMLE` markdown still does **not** fully document many structure-level ideas as formal release-backlog items, including:

- full prior blending design details,
- rerank design details,
- dense vs sparse prior execution paths,
- explicit `relative_bias` / `rope` position backlog,
- explicit `embedding_fusion` variants,
- explicit `share_blocks` variants,
- explicit conv / sequence-mixer extensions,
- explicit gated prior / prior distillation / two-stage rerank design space.

So the answer is:

- **Yes**, `VirtualMLE` markdown already includes some structural-improvement hints.
- **But no**, it does not yet contain the full detailed improvement menu that exists in the main experimental branch and related Baby reflections.

## 7. Recommended interpretation

If the goal is to keep `VirtualMLE` as a clean release baseline, the current markdown coverage is reasonable.

If the goal is to turn `VirtualMLE` into a stronger experimental release branch, the next missing documentation layers would be:

1. prior blending design,
2. rerank design,
3. position / pooling / FFN / sharing structural variants,
4. train-time prior distillation and gating ideas.
