"""Build IOP-compliant vector figures + LaTeX table numbers from the FULL-run results."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.stats import spearmanr

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.5,
})
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "green": "#008300",
     "violet": "#6250d6", "red": "#e34948", "grey": "#6b6b66"}

M = pd.read_csv(f"{RES}/all_metrics_long.csv")
A = pd.read_csv(f"{RES}/all_accuracy.csv")
S2 = pd.read_csv(f"{RES}/all_ood_set_stats_v2.csv")
S1 = pd.read_csv(f"{RES}/all_ood_set_stats.csv")
REC = pd.read_csv(f"{RES}/recurrence_curves.csv")
RECS = pd.read_csv(f"{RES}/recurrence_summary.csv")
DS = ["synth2d", "wine_red_white", "covertype", "digits_split", "mnist_fmnist"]
DSN = {"synth2d": "Two moons (2-D)", "wine_red_white": "Wine", "covertype": "Covertype",
       "digits_split": "Digits", "mnist_fmnist": "MNIST"}
NUM = {}

# ---------------- Fig 2: AUROC vs translation ----------------
models = [("qnn", "sym", "msp", "QNN [−π, π]", C["blue"], "-", "o"),
          ("qnn", "half", "msp", "QNN [0, π]", C["orange"], "--", "s"),
          ("qnn_ens", "half", "ens_var", "QNN ensemble [0, π]", C["aqua"], "-.", "^"),
          ("qnn_hybrid", "sym", "msp", "QNN + hybrid head", C["violet"], ":", "D"),
          ("rff", "sym", "msp", "RFF (same spectrum)", C["red"], (0, (5, 1, 1, 1)), "v"),
          ("knn", "sym", "dist", "kNN distance", C["grey"], "-", "x")]
fig, axes = plt.subplots(1, 5, figsize=(15 * CM, 5.2 * CM), sharey=True)
fig.subplots_adjust(wspace=0.12)
for ax, ds in zip(axes, DS):
    D = M[(M.dataset == ds) & M.ood_set.str.startswith("translate_")].copy()
    D["t"] = D.ood_set.str.split("_").str[1].astype(float)
    for m, sc, s, lab, col, ls, mk in models:
        E = D[(D.model == m) & (D.scaling == sc) & (D.score == s)].groupby("t").auroc.agg(["mean", "std"])
        ax.plot(E.index, E["mean"], ls=ls, marker=mk, ms=3.2, lw=1.1, color=col, label=lab)
    ax.axhline(0.5, color="#999", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.tick_params(axis="x", which="minor", length=0)
    ax.set_title(DSN[ds]); ax.set_xlabel("shift (σ)")
    ax.set_ylim(0.25, 1.02)
axes[0].set_ylabel("AUROC")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.33), handlelength=2.8, columnspacing=1.4)
fig.savefig(f"{OUT}/figure4.pdf"); plt.close(fig)

# ---------------- Fig 3: corrected alias frequency per dataset ----------------
nat2 = S2[~S2.ood_set.str.startswith("alias")]
fig, ax = plt.subplots(figsize=(8.5 * CM, 5.0 * CM))
x = np.arange(len(DS)); w = 0.36
for i, (sc, lab, col, hatch) in enumerate([("half", "min–max to [0, π]", C["orange"], "////"),
                                            ("sym", "min–max to [−π, π]", C["blue"], "")]):
    g = nat2[nat2.scaling == sc].groupby("dataset").frac_aliased_v2
    mean, mx = g.mean().reindex(DS) * 100, g.max().reindex(DS) * 100
    bars = ax.bar(x + (i - 0.5) * w, mean, w * 0.92, color=col, alpha=0.85, hatch=hatch, edgecolor="white",
                  linewidth=0.6, label=f"{lab} (mean)")
    ax.scatter(x + (i - 0.5) * w, mx, marker="_" if i else "_", s=60, color="k", lw=1.2, zorder=3,
               label="max over shift sets" if i else None)
    for xi, v in zip(x + (i - 0.5) * w, mean):
        if v >= 0.05: ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    NUM[f"alias_mean_{sc}"] = {d: round(float(v), 2) for d, v in mean.items()}
    NUM[f"alias_max_{sc}"] = {d: round(float(v), 2) for d, v in mx.items()}
ax.set_xticks(x); ax.set_xticklabels(["Two moons\n(d = 2)", "Wine\n(d = 8)", "Covertype\n(d = 8)", "Digits\n(d = 8)", "MNIST\n(d = 8)"], fontsize=6.8)
ax.set_ylabel("shifted points aliased (%)"); ax.set_ylim(0, 85)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
fig.savefig(f"{OUT}/figure2.pdf"); plt.close(fig)

# ---------------- Fig 4: paired sym vs half ----------------
J = M[(M.model == "qnn") & (M.score == "msp")].merge(S2, on=["dataset", "seed", "scaling", "ood_set"])
Jn = J[~J.ood_set.str.startswith("alias")]
P = Jn.pivot_table(index=["dataset", "seed", "ood_set"], columns="scaling", values=["auroc", "frac_aliased_v2"])
dd = pd.DataFrame({"dal": (P["frac_aliased_v2"]["sym"] - P["frac_aliased_v2"]["half"]) * 100,
                   "dau": P["auroc"]["sym"] - P["auroc"]["half"]}).reset_index()
r = spearmanr(dd.dal, dd.dau)
NUM["paired"] = dict(rho=round(float(r.correlation), 3), p=float(r.pvalue), n=int(len(dd)),
                     big_n=int((dd.dal > 10).sum()), big_dau=round(float(dd[dd.dal > 10].dau.mean()), 3),
                     small_dau=round(float(dd[dd.dal <= 10].dau.mean()), 3),
                     big_synth=int(((dd.dal > 10) & (dd.dataset == "synth2d")).sum()))
fig, ax = plt.subplots(figsize=(8.5 * CM, 5.6 * CM))
mk = {"synth2d": ("o", C["blue"]), "wine_red_white": ("s", C["orange"]), "covertype": ("^", C["aqua"]),
      "digits_split": ("D", C["violet"]), "mnist_fmnist": ("v", C["red"])}
for ds in DS:
    e = dd[dd.dataset == ds]
    ax.scatter(e.dal, e.dau, marker=mk[ds][0], s=14, facecolor="none", edgecolor=mk[ds][1], lw=0.9, label=DSN[ds])
ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6)
ax.set_xlabel("Δ aliased fraction, [−π, π] − [0, π] (points, %)")
ax.set_ylabel("Δ AUROC, [−π, π] − [0, π]")
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
fig.savefig(f"{OUT}/figure3.pdf"); plt.close(fig)

# ---------------- Fig 5: recurrence ----------------
fig, axes = plt.subplots(1, 2, figsize=(15 * CM, 4.6 * CM), sharey=True)
T = REC["T"].values / (2 * np.pi)
for ax, ds in zip(axes, ["wine_red_white", "digits_split"]):
    ax.plot(T, REC[f"{ds}|integer|axis0"], color=C["blue"], lw=0.9, label="integer frequencies")
    ax.plot(T, REC[f"{ds}|incommensurate|axis0"], color=C["orange"], lw=0.9, ls="--", label="trainable incommensurate")
    ax.set_title({"wine_red_white":"(a) Wine","digits_split":"(b) Digits"}[ds])
    ax.set_xlabel("shift T / 2π")
axes[0].set_ylabel("D(T)")
axes[0].legend(frameon=False, loc="upper right")
fig.savefig(f"{OUT}/figure5.pdf"); plt.close(fig)

# ---------------- numbers for tables ----------------
own = M[((M.scaling == "sym") & (M.ood_set == "alias_sym")) | ((M.scaling == "half") & (M.ood_set == "alias_half")) |
        ((M.scaling == "squash") & (M.ood_set == "alias_sym"))]
t1 = own.groupby(["model", "scaling", "score"]).auroc.agg(["mean", "min", "max", "count"]).round(4)
NUM["table1"] = t1.reset_index().to_dict("records")
acc = A.groupby(["dataset", "model", "scaling"]).accuracy.agg(["mean", "std"]).reset_index()
NUM["accuracy"] = acc.round(4).to_dict("records")
nat = M[~M.ood_set.str.startswith("alias")]
best = nat.groupby(["dataset", "model", "scaling", "score"]).auroc.mean().reset_index()
b = best.loc[best.groupby(["dataset", "model", "scaling"]).auroc.idxmax()]
NUM["natural_best"] = b.round(4).to_dict("records")
far = nat[nat.ood_set.str.startswith("far")].groupby(["dataset", "model", "scaling", "score"]).agg(
    auroc=("auroc", "mean"), fpr=("fpr95", "mean")).reset_index()
NUM["far"] = far.round(4).to_dict("records")
cov = S1.groupby(["dataset", "scaling"]).torus_coverage.mean()
NUM["coverage"] = {f"{d}|{s}": round(float(v), 4) for (d, s), v in cov.items()}
part = {}
for ds in DS:
    jj = Jn[Jn.dataset == ds]
    X = np.c_[np.ones(len(jj)), jj.frac_out_of_range.rank()]
    ra, rb = jj.frac_aliased_v2.rank(), jj.auroc.rank()
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]; eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    part[ds] = round(float(np.corrcoef(ea, eb)[0, 1]), 3)
NUM["partial"] = part
ra = spearmanr(Jn.frac_aliased_v2, Jn.auroc); NUM["overall_rho"] = [round(float(ra.correlation), 3), float(ra.pvalue)]
NUM["recurrence"] = RECS.drop(columns=[c for c in ["note"] if c in RECS]).round(3).to_dict("records")
json.dump(NUM, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json"), "w"), indent=1, default=str)
print("ok")
