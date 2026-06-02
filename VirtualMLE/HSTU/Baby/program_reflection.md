# program_reflection — HSTU / Baby

> For the release baseline, keep the evaluation protocol consistent across runs and do not use the test set as a tuning signal.

> This file defines the reflection-driven experiment workflow for the current release cell. Before each run, write down the current best setting, the remaining coverage gap, and the single primary hypothesis of the next run. Keep/discard decisions must be made from validation metrics only; test metrics should be read once at the very end for the validation-best configuration.

## 1. Goal

Improve validation NDCG@10 on `HSTU` for the `Baby` domain while keeping the release code path simple, reproducible, and easy to audit.

## 2. Startup Checks

Before starting a new round or resuming a search, always check:

1. `../run_hstu.py`
2. `program_reflection.md`
3. `sequential_data_processed.txt`

Then confirm:

- `../run_hstu.py` can resolve `--domain baby` to the default dataset path.
- The dataset file exists and is readable.
- The first round of any fresh search is a plain baseline run with no code modification.

## 3. Working Files

This release cell should keep only a minimal set of active files:

- `program_reflection.md`: this workflow definition.
- `sequential_data_processed.txt`: the release-ready sequence dataset.
- `output/`: logs and JSON summaries produced by actual runs.

Notes:

- Do not edit the dataset file itself.
- Do not fabricate JSON results manually.
- Do not use intermediate test metrics for keep/discard decisions.

## 4. Run Protocol

Run inside the current dataset directory and save both logs and JSON summaries explicitly:

```bash
mkdir -p output
python ../run_hstu.py   --domain baby   --output_json output/run_result_<tag>.json   > output/run_<tag>.log 2>&1
```

If a concrete change is being tested, append only the required arguments in the same command, for example:

```bash
python ../run_hstu.py   --domain baby   --epochs 20   --hidden_dim 128   --linear_dim 16   --attention_dim 16   --num_blocks 2   --num_heads 2   --dropout 0.3   --attn_dropout 0.1   --optimizer adamw   --lr 0.001   --weight_decay 0.0   --temperature 0.05   --output_json output/run_result_search_<tag>.json   > output/run_search_<tag>.log 2>&1
```

## 5. What Must Be Written Before Each Run

Before launching a run, record at least the following in your short-term working note or session context:

1. The current best retained configuration and its primary validation metric.
2. The uncovered or under-covered directions in the current backlog.
3. The single primary hypothesis of this round.
4. The expected gain mechanism and the main risk.
5. The rollback or follow-up plan if the run fails.

## 6. Experiment Loop

Each round should start from the current best retained configuration and follow this loop:

1. Choose one clear and testable change based on the current coverage gap.
2. Modify `../run_hstu.py` only when a genuinely new configurable switch is needed; otherwise keep the code fixed and change only runtime arguments.
3. Run the experiment with a unique log / JSON tag.
4. Read validation metrics and parameter count from the JSON summary first.
5. If the run fails because of an implementation issue, fix the low-level bug and rerun quickly; if the idea itself is unstable or clearly unsuitable, mark it as discard/crash and roll back.
6. Keep or discard the change using validation metrics only.
7. After each round, return to the top of the loop automatically unless a human explicitly asks to stop.

Default keep/discard rule:

- Primary metric: `val NDCG@10`.
- Higher primary validation metric: `keep`.
- Lower primary validation metric: `discard`.
- If the primary metric is effectively tied, prefer the simpler and more stable implementation; otherwise discard.

## 7. Coverage Requirements

The purpose of reflection is not only to find a local best run, but also to rule out other reasonable directions systematically. At regular intervals, check what has not been covered yet.

- `hidden_dim`, `linear_dim`, `attention_dim`, `num_blocks`, `num_heads`, `dropout`, `attn_dropout`, `batch_size`, `eval_batch_size`, `lr`, `weight_decay`, `optimizer`, `temperature`, `train_targets`, `train_window_size`, `early_stop_patience` are all part of the formal search space.
- `num_negatives` should stay fixed at the script default `500`; do not treat it as a tuning axis in the release baseline.
- Coverage matters more than greedy local tuning: if too many rounds stay on only one or two axes, force a new uncovered direction into the queue.
- Keep the release baseline simple and auditable; if a new structural switch is added later, expose it explicitly in `../run_hstu.py` before running experiments.

## 8. Suggested Result Table Schema

If a structured TSV is maintained later, the recommended header is:

```tsv
run_id	commit	model	domain	hypothesis	changed_dimension	model_params_m	val_recall@5	val_recall@10	val_ndcg@5	val_ndcg@10	status	failure_dimension	reflection	generalized_rule	next_action
```

Recording rules:

- `model_params_m` should be computed from `model_stats.total_params / 1e6` and rounded to 3 decimals.
- Validation metrics should keep 6 decimals when possible.
- Test metrics do not belong in the middle of the search table.

## 9. Final Reporting

At the end of the search, select the configuration with the best validation `NDCG@10` and read `test_metrics` once for that validation-best configuration only.

## 10. Structural / Experimental Backlog

These are formal backlog directions for the release baseline, not optional afterthoughts:

1. Backbone scale: compare `hidden_dim`, `linear_dim`, `attention_dim`, `num_blocks`, and `num_heads` in a controlled way and always record parameter count together with validation metrics.
2. Optimization behavior: compare `AdamW` vs `Adam`, then refine `lr`, `weight_decay`, and `temperature` around the stronger setting.
3. Training supervision: compare `all_positions` vs `last_position` and verify whether gains come from richer supervision or from a cleaner optimization target.
4. Regularization and temporal mixing strength: test `dropout`, `attn_dropout`, and history truncation without changing multiple structure axes in the same round.

## 11. Current Domain Facts

- Number of users: 19445
- Number of items: 7050
- Average full-sequence length: 8.24
- P50 / P90 / P95 sequence length: 6 / 13 / 17
- Initial reading: short-sequence dominated

## 12. Transfer Lessons for Baby

The points below summarize practical transfer lessons distilled from Baby-domain search history and from cross-domain evidence that remained useful after validation on Baby. Treat them as prioritized hypotheses for future runs, not as substitutes for validation.

1. **Short-sequence Baby favors disciplined regularization before aggressive capacity growth.**
   - With average sequence length below 10, stronger regularization was often more helpful than immediately increasing depth or freeing more parameters.
   - Search `dropout`, supervision style, and embedding constraints early before moving to larger backbones.

2. **Shared item embeddings are a strong prior in Baby.**
   - Untying the input/output item tables increased parameters substantially but hurt validation quality.
   - For Baby-style short sequences, tying is not only a memory-saving choice; it behaves like a useful inductive bias.

3. **Do not assume embedding normalization is always beneficial.**
   - Disabling embedding L2 normalization improved Baby validation performance in prior runs.
   - This suggests that Baby can benefit from preserving embedding magnitude information instead of enforcing unit-norm embeddings by default.

4. **Keep multi-position supervision unless strong evidence says otherwise.**
   - Moving from `all_positions` to `last_position` sharply reduced the number of supervision targets and degraded Baby validation performance.
   - On short-sequence domains, multi-position supervision is an important sample-efficiency prior.

5. **Attention dropout should be tested conservatively.**
   - Baby did not benefit from large attention-level dropout in the earlier search history.
   - If attention dropout is revisited, start from very small values and treat zero or near-zero as a serious baseline.

6. **A moderate temperature and reasonably strong dropout transferred better than extreme values.**
   - Baby was sensitive to both over-sharpened and over-smoothed logits.
   - When porting settings from another domain, treat temperature and dropout as Baby-specific recalibration knobs rather than fixed global constants.

7. **Train-split co-occurrence priors are a legitimate high-value direction for Baby.**
   - Row-normalized model scores combined with train-split `transition / itemCF / seqCF` priors formed one of the most stable legal gain paths.
   - Keep the prior scope on the training split for comparable experiments; use full-sequence priors only as diagnostic upper bounds.

8. **If a new pooling or reranking idea fails, first check whether it diluted recency.**
   - In short-sequence recommendation, the most recent action carries disproportionate signal.
   - Failures of smoothing-style heads often come from weakening recency, not from insufficient model size.

9. **For Baby, protocol correctness matters as much as model choice.**
   - Evaluation masking should preserve the current target even if it also appeared in history.
   - Mistakes in this detail can distort Recall/NDCG substantially in repeat-purchase short-sequence domains.
