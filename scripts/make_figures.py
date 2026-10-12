#!/usr/bin/env python3
"""Render the ESM-2 vs ESM-C paper figures (SVG only) and tables.

All Spearman numbers recompute from figure_lib on the cached embeddings and are
memoized to figures/data/spearman_sweep.json so panels, tables, and reruns share
one computation (TopLayerRegressor is ~1.5 s/fit, so caching matters). Plate
numbers come from scripts/plate_selection.py output. Vector SVG only, no PNG.
Colors are the single-hue light/dark blue pair (ESM-2 light, ESM-C dark).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; SVG output, never PNG
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_lib as fl  # noqa: E402
from plm_evolve.variants import read_wt_sequence  # noqa: E402
from validate_gap2_dms import PRESETS  # noqa: E402

ROOT = fl._ROOT
BUNDLE = ROOT / "figures" / "esm2_vs_esmc"   # all paper artifacts gathered here
DATA = BUNDLE / "data"
FIG1_PATH = BUNDLE / "fig1_esmc_vs_esm2.svg"
FIGS1_PATH = BUNDLE / "figS1_size_dims.svg"
FIGS2_PATH = BUNDLE / "figS2_pooling_null.svg"
TABLES_PATH = BUNDLE / "tables.md"
TRAIN_SIZES = [16, 32, 64, 95, 190]
SEEDS = 20

# ----- memoized Spearman sweep (in-memory + on-disk) -------------------------
_ALIGNED: dict = {}
_SWEEP_PATH = DATA / "spearman_sweep.json"
_sweep: dict = {}
_sweep_dirty = False


def _load_sweep():
    global _sweep
    if _SWEEP_PATH.exists():
        _sweep = json.loads(_SWEEP_PATH.read_text())


def _aligned(dataset, hash_, pooling, model):
    key = (dataset, hash_, pooling, model)
    if key not in _ALIGNED:
        _ALIGNED[key] = fl.load_aligned(dataset, [model], hash_, pooling=pooling)
    return _ALIGNED[key]


def _cell(dataset, model, hash_, t, pooling="mean"):
    """Memoized {mean, se, samples, width} for one (dataset, model, pooling, t)."""
    global _sweep_dirty
    key = f"{dataset}|{model}|{pooling}|{t}"
    if key in _sweep:
        return _sweep[key]
    ids, y, X = _aligned(dataset, hash_, pooling, model)
    s = fl.held_out_spearman_samples(X[model], y, [t], seeds=SEEDS)[t]
    rec = {
        "mean": float(s.mean()),
        "se": float(s.std(ddof=0) / np.sqrt(len(s))),
        "samples": [float(v) for v in s],
        "width": int(X[model].shape[1]),
        "n": int(len(y)),
    }
    _sweep[key] = rec
    _sweep_dirty = True
    return rec


def _flush_sweep():
    if _sweep_dirty:
        DATA.mkdir(parents=True, exist_ok=True)
        _SWEEP_PATH.write_text(json.dumps(_sweep, indent=0))


def _spear(dataset, model, t, pooling="mean"):
    """hash is the medium hash unless the model is a matched-pair model."""
    hash_ = fl.DATASET_HASH[dataset]
    return _cell(dataset, model, hash_, t, pooling)


_load_sweep()
E2, EC = fl.MEDIUM_PAIR
E2_BIG, EC_BIG = fl.MATCHED_PAIR
E2H, ECH = fl.HEADLINE_PAIR  # headline ESM-2 650M vs ESM-C 300M (panels a/b/c, Table 1)

# Shared style. svg.fonttype="none" keeps text as text so the SVG references the
# font by name (Myriad Pro renders wherever the font is installed, e.g. Illustrator).
FONT = ["Myriad Pro", "sans-serif"]
BAR_EDGE = dict(edgecolor="black", linewidth=0.6)


def _style():
    plt.rcParams.update({"font.family": FONT, "svg.fonttype": "none"})


# ----- Figure 1 --------------------------------------------------------------
def build_fig1(out_path):
    _style()
    fig = plt.figure(figsize=(7.0, 6.0))  # width well under A4 (11.7in landscape)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.6, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    w = 0.38

    # (a) per-protein held-out Spearman at train=95, paired bars + SE
    e2 = [_spear(d, E2H, 95)["mean"] for d in fl.PROTEIN_ORDER]
    ec = [_spear(d, ECH, 95)["mean"] for d in fl.PROTEIN_ORDER]
    e2se = [_spear(d, E2H, 95)["se"] for d in fl.PROTEIN_ORDER]
    ecse = [_spear(d, ECH, 95)["se"] for d in fl.PROTEIN_ORDER]
    x = np.arange(len(fl.PROTEIN_ORDER))
    ax_a.bar(x - w / 2, e2, w, yerr=e2se, capsize=2, label=fl.LABELS[E2H], color=fl.ESM2_COLOR, **BAR_EDGE)
    ax_a.bar(x + w / 2, ec, w, yerr=ecse, capsize=2, label=fl.LABELS[ECH], color=fl.ESMC_COLOR, **BAR_EDGE)
    ax_a.axhline(0, color="0.6", lw=0.8)
    ax_a.set_xticks(x); ax_a.set_xticklabels(fl.PROTEIN_ORDER)
    ax_a.set_ylim(0, 0.85)  # headroom so the legend and inset clear the bars
    ax_a.set_ylabel("held-out Spearman (train=95)")
    ax_a.legend(frameon=False, loc="upper left", fontsize=8)
    def _esmc_sig_win(d):
        e = _spear(d, E2H, 95); c = _spear(d, ECH, 95)
        diff = np.array(c["samples"]) - np.array(e["samples"])
        return diff.mean() > 2 * (diff.std(ddof=0) / np.sqrt(len(diff)))
    _wins = sum(1 for d in fl.PROTEIN_ORDER if _esmc_sig_win(d))
    ax_a.set_title(f"Smaller ESM-C 300M outranks the larger ESM-2 650M ({_wins} of {len(fl.PROTEIN_ORDER)} proteins)", fontsize=9)
    ax_a.text(-0.05, 1.06, "a", transform=ax_a.transAxes, fontweight="bold", fontsize=13)

    # (b) learning curve: Spearman vs train size, kelsic + cas12f
    for ds, ls in (("kelsic", "-"), ("cas12f", "--")):
        for m, col in ((E2H, fl.ESM2_COLOR), (ECH, fl.ESMC_COLOR)):
            ys = np.array([_spear(ds, m, t)["mean"] for t in TRAIN_SIZES])
            es = np.array([_spear(ds, m, t)["se"] for t in TRAIN_SIZES])
            ax_b.plot(TRAIN_SIZES, ys, ls, color=col, marker="o", ms=3,
                      label=f"{fl.LABELS[m]} ({ds})")
            ax_b.fill_between(TRAIN_SIZES, ys - es, ys + es, color=col, alpha=0.15)
    for xv in (95, 190):
        ax_b.axvline(xv, color="0.7", lw=0.7, ls=":")
    ylo = ax_b.get_ylim()[0]
    ax_b.annotate("1 plate", (95, ylo), fontsize=6, color="0.5", ha="center")
    ax_b.annotate("2 plates", (190, ylo), fontsize=6, color="0.5", ha="center")
    ax_b.set_xlabel("train size (variants measured)"); ax_b.set_ylabel("held-out Spearman")
    ax_b.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax_b.text(-0.18, 1.04, "b", transform=ax_b.transAxes, fontweight="bold", fontsize=13)

    # (c) plate-2 enrichment over baseline, all 5 proteins, paired bars + delta
    plate = json.loads((DATA / "plate_precision.json").read_text())
    order = fl.PROTEIN_ORDER
    e2_enr = [plate[d][fl.LABELS[E2H]]["enrichment"] for d in order]
    ec_enr = [plate[d][fl.LABELS[ECH]]["enrichment"] for d in order]
    xx = np.arange(len(order))
    ax_c.bar(xx - w / 2, e2_enr, w, color=fl.ESM2_COLOR, label=fl.LABELS[E2H], **BAR_EDGE)
    ax_c.bar(xx + w / 2, ec_enr, w, color=fl.ESMC_COLOR, label=fl.LABELS[ECH], **BAR_EDGE)
    for i, d in enumerate(order):
        ax_c.text(i, max(e2_enr[i], ec_enr[i]) + 0.6, f"+{plate[d]['delta']:.1f}",
                  ha="center", fontsize=6, color="0.3")
    ax_c.axhline(0, color="0.6", lw=0.8)
    ax_c.set_xticks(xx); ax_c.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
    ax_c.set_ylabel("plate-2 enrichment\n(precision@95 - random)")
    ax_c.text(-0.18, 1.04, "c", transform=ax_c.transAxes, fontweight="bold", fontsize=13)

    for ax in (ax_a, ax_b, ax_c):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "fig1_numbers.json").write_text(json.dumps(
        {"train95": {d: {"esm2": e2[i], "esmc": ec[i], "esm2_se": e2se[i], "esmc_se": ecse[i]}
                     for i, d in enumerate(fl.PROTEIN_ORDER)}}, indent=2))


# ----- Figure S1 (size, dimensionality) --------------------------------------
_S1_MODELS = [E2, E2_BIG, EC, EC_BIG]  # 150M, 650M, 300M, 600M


def build_figS1(out_path):
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))
    vals = {m: _spear("kelsic", m, 64) for m in _S1_MODELS}

    # (a) Spearman vs model size, family-colored -- why ESM-C 300M is the chosen scale
    labels = [fl.LABELS[m] for m in _S1_MODELS]
    cols = [fl.ESM2_COLOR, fl.ESM2_COLOR, fl.ESMC_COLOR, fl.ESMC_COLOR]
    ax1.bar(range(4), [vals[m]["mean"] for m in _S1_MODELS],
            yerr=[vals[m]["se"] for m in _S1_MODELS], capsize=3, color=cols, **BAR_EDGE)
    ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax1.set_ylabel("held-out Spearman (kelsic, train=64)")
    ax1.set_title("Bigger ESM-2 does not help; ESM-C 300M matches 600M", fontsize=7.5)
    ax1.text(-0.16, 1.05, "a", transform=ax1.transAxes, fontweight="bold", fontsize=13)

    # (b) Spearman vs embedding dimensionality, family-colored
    for m, col in zip(_S1_MODELS, cols):
        ax2.scatter(vals[m]["width"], vals[m]["mean"], color=col, s=40, zorder=3)
        ax2.annotate(fl.LABELS[m], (vals[m]["width"], vals[m]["mean"]),
                     fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax2.set_xlabel("embedding dimensionality")
    ax2.set_ylabel("held-out Spearman (kelsic, train=64)")
    ax2.set_title("Edge is not feature count", fontsize=8)
    ax2.text(-0.18, 1.05, "b", transform=ax2.transAxes, fontweight="bold", fontsize=13)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# ----- Figure S2 (pooling null) ----------------------------------------------
# Pairs with BOTH mean and mutated_token caches (cas12f esm2-150M has no mutated
# cache, so the ESM-2 examples use kelsic 150M/650M and the cross-protein ESM-C
# check uses cas12f 300M).
_S2_PAIRS = [("kelsic", E2), ("kelsic", E2_BIG), ("kelsic", EC),
             ("kelsic", EC_BIG), ("cas12f", EC)]


def build_figS2(out_path):
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    w = 0.38
    means = {pool: [] for pool in ("mean", "mutated_token")}
    ses = {pool: [] for pool in ("mean", "mutated_token")}
    xlabels = []
    for ds, m in _S2_PAIRS:
        xlabels.append(f"{ds}\n{fl.LABELS[m]}")
        for pool in ("mean", "mutated_token"):
            c = _spear(ds, m, 64, pooling=pool)
            means[pool].append(c["mean"]); ses[pool].append(c["se"])
    x = np.arange(len(_S2_PAIRS))
    ax.bar(x - w / 2, means["mean"], w, yerr=ses["mean"], capsize=2,
           color="0.45", label="mean pool", **BAR_EDGE)
    ax.bar(x + w / 2, means["mutated_token"], w, yerr=ses["mutated_token"], capsize=2,
           color=fl.ESMC_COLOR, label="mutated-token pool", **BAR_EDGE)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=6.5)
    ax.set_ylabel("held-out Spearman (train=64)")
    ax.set_title("Mutated-token pooling does not beat mean (GAP-2 null)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# ----- Tables ----------------------------------------------------------------
def _length(dataset):
    return len(read_wt_sequence(PRESETS[dataset].wt_fasta))


def build_tables(out_path=None):
    out_path = out_path or TABLES_PATH
    plate = json.loads((DATA / "plate_precision.json").read_text())
    L = []
    L.append("# ESM-2 vs ESM-C comparison tables")
    L.append("")
    L.append("Generated by scripts/make_figures.py:build_tables from the recomputed")
    L.append("Spearman sweep and scripts/plate_selection.py output. Held-out Spearman is")
    L.append("the paired metric (RandomForest top-layer on mean-pool embeddings, 20 seeds);")
    L.append("plate-2 enrichment is precision@95 minus the random baseline 95^2/(n-95).")
    L.append("")
    L.append("## Table 1. Per-protein ranking and per-plate enrichment (ESM-2 650M vs ESM-C 300M)")
    L.append("")
    L.append("| protein | L | n | ESM-2 Spearman (ts=95) | ESM-C Spearman (ts=95) | gap | >2 SE | plate baseline | ESM-2 prec / enr | ESM-C prec / enr | delta |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in fl.PROTEIN_ORDER:
        e2 = _spear(d, E2H, 95); ec = _spear(d, ECH, 95)
        diff = np.array(ec["samples"]) - np.array(e2["samples"])
        gap = float(diff.mean()); gse = float(diff.std(ddof=0) / np.sqrt(len(diff)))
        sep = "yes" if abs(gap) > 2 * gse else "no"
        p = plate[d]; p2 = p[fl.LABELS[E2H]]; pc = p[fl.LABELS[ECH]]
        L.append(f"| {d} | {_length(d)} | {p['n']} | "
                 f"{e2['mean']:.3f} +/- {e2['se']:.3f} | {ec['mean']:.3f} +/- {ec['se']:.3f} | "
                 f"{gap:+.3f} | {sep} | {p['baseline']:.1f} | "
                 f"{p2['precision_at_95']:.1f} / {p2['enrichment']:+.1f} | "
                 f"{pc['precision_at_95']:.1f} / {pc['enrichment']:+.1f} | {p['delta']:+.1f} |")
    L.append("")
    mk2 = _spear("kelsic", E2_BIG, 64); mkc = _spear("kelsic", EC_BIG, 64)
    L.append(f"Matched scale (kelsic, train=64): ESM-2 650M {mk2['mean']:.3f} vs "
             f"ESM-C 600M {mkc['mean']:.3f}. enrichment = precision@95 minus the random "
             f"baseline 95^2/(n-95). gap = ESM-C minus ESM-2 paired over seeds; >2 SE marks "
             f"the gap as separated. Raw precision@95 is not comparable across proteins "
             f"(baseline differs with n); enrichment and delta are.")
    L.append("")
    L.append("## Table S1. Full held-out Spearman sweep (mean +/- SE, 20 seeds)")
    L.append("")
    L.append("| protein | model | train size | Spearman |")
    L.append("|---|---|---|---|")
    rows = [("kelsic", E2), ("kelsic", EC), ("stiffler", E2), ("stiffler", EC),
            ("giacomelli", E2), ("giacomelli", EC), ("cas12f", E2), ("cas12f", EC),
            ("brenan", E2), ("brenan", EC), ("kelsic", E2_BIG), ("kelsic", EC_BIG)]
    for d, m in rows:
        for t in TRAIN_SIZES:
            c = _spear(d, m, t)
            L.append(f"| {d} | {fl.LABELS[m]} | {t} | {c['mean']:.3f} +/- {c['se']:.3f} |")
    L.append("")
    L.append("Plate-2 precision@95 and enrichment per protein are in Table 1; the pooling")
    L.append("null (mean vs mutated-token) is Figure S2.")
    L.append("")
    text = "\n".join(L)
    assert chr(0x2014) not in text, "em-dash in generated tables"
    for bad in ("results", "discussion", "conclusion"):
        import re
        assert not re.search(rf"\b{bad}\b", text, re.I), f"forbidden word {bad}"
    out_path.write_text(text)
    return out_path


def build_all():
    _load_sweep()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    build_fig1(FIG1_PATH); _flush_sweep(); print("fig1 done", flush=True)
    build_figS1(FIGS1_PATH); _flush_sweep(); print("figS1 done", flush=True)
    build_figS2(FIGS2_PATH); _flush_sweep(); print("figS2 done", flush=True)
    build_tables(); _flush_sweep(); print("tables done", flush=True)
    print("wrote fig1, figS1, figS2, tables; sweep cells:", len(_sweep), flush=True)


if __name__ == "__main__":
    build_all()
