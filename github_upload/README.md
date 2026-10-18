# Exact aliasing blind spots in OOD detection with data re-uploading QNNs

Code and results for the paper:

> A. Tripathi, A. Rasool and N. Khare, "Exact aliasing blind spots in out-of-distribution
> detection with data re-uploading quantum neural networks" (submitted to *Physica Scripta*).

## Contents
- `qnn_aliasing_ood_v2.ipynb` — runs every experiment (exact NumPy state-vector QNN simulator
  with adjoint gradients, remedies, baselines, OOD scores) and zips all results.
- `results/` — per-run metrics, per-sample OOD scores (`scores/*.npz`), summary tables, figures.
- `manuscript/make_figures.py` — rebuilds the paper figures from `results/`.

## Reproduce
```bash
pip install -r requirements.txt
jupyter notebook qnn_aliasing_ood_v2.ipynb   # set MODE = "FULL", then Run All (~3 CPU-hours)
```
Datasets (Wine, MNIST, Fashion-MNIST via OpenML; Covertype and Digits via scikit-learn) download
automatically on first run. All runs are deterministic: seeds 0, 1, 2.

## License
Code: MIT. Results data: CC BY 4.0.
