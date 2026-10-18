# Results: aliasing blind spots in re-uploading QNNs

Generated 2026-10-18T14:08:21  |  MODE=FULL

Datasets run: ['covertype', 'digits_split', 'mnist_fmnist', 'synth2d', 'wine_red_white']  |  skipped: {}

## Files
- `config.json` — every hyperparameter used.
- `sanity_checks.json` — adjoint-gradient error vs finite differences; max output change under exact 2πk shifts.
- `all_metrics_long.csv` — one row per (dataset, seed, model, scaling, score, ood_set): AUROC, FPR@95.
- `summary_auroc_fpr95.csv` — mean/std over seeds.
- `all_accuracy.csv`, `table4_accuracy.csv` — ID test accuracy.
- `all_ood_set_stats.csv`, `table3_alias_frequency.csv` — v1 alias metric (overcounts points just past the boundary; superseded).
- `all_ood_set_stats_v2.csv`, `table3b_alias_frequency_corrected.csv` — corrected alias metric: near the data only via wrap-around.
- `table1_exact_alias_auroc.csv` — theorem check: periodic models must score 0.5 on `alias_*` sets.
- `table2_tradeoff.csv` — mean AUROC over shift/far sets vs accuracy.
- `recurrence_summary.csv`, `recurrence_curves.csv` — E4 quasi-periodic recurrence.
- `scores/<dataset>_seed<k>.npz` — per-sample OOD scores; keys `model|scaling|score|set` (set `ID` = in-distribution test).
- `figures/` — fig1 confidence maps, fig2 AUROC vs translation, fig3 alias fraction vs AUROC, fig4 trade-off, fig5 recurrence, fig6 alias heatmap.
- `run_log.txt`, `environment.json`.

## Naming
- scaling: `sym` = min-max to [-π, π]; `half` = [0, π]; `squash` = (π/2)·tanh(z/s); classical baselines use `sym` (affine, no wrapping).
- models: `qnn` (integer freq.), `qnn_ens` (ensemble), `qnn_gdrop` (Gaussian parameter noise, MC), `qnn_incomm` (trainable incommensurate freq.),
  `qnn_squash`, `qnn_hybrid` (distance-gated logits), `rff` (random Fourier features on the QNN integer spectrum), `mlp`, `mlp_ens`, `svm`, `gp`, `knn` (distance only).
- scores (higher = more OOD): `msp` = −max prob, `energy` = −logsumexp(logits), `ens_var`, `mc_var`, `grad` = per-sample parameter-gradient norm (extrapolation-score proxy), `dist` = kNN distance.
- OOD sets: `translate_t` (shift by t training std, random sign per feature), `scale_s` (μ + s(x−μ)), `far:*` (other dataset/classes),
  `alias_sym` / `alias_half` (exact lattice shifts u + 2πk for that scaling, mapped back to feature space).