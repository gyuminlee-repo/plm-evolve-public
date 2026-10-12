#!/usr/bin/env python3
"""Run FolDE's actual predictors on plm-evolve's cached embeddings + identical
held-out splits, computing the SAME held-out Spearman as the figures. Puts FolDE
as a third arm next to the EVOLVEpro-core RF.

Arms:
  FolDE-RF      = RandomForestFewShotModel (sanity: must match plm-evolve RF)
  FolDE-BT-MLP  = TorchMLPFewShotModel, Bradley-Terry ranking loss, no warm-start
"""
import os
import sys

WS = os.environ["WORKSPACE_ROOT"]
sys.path.insert(0, os.path.join(WS, "cc", "plm-evolve", "scripts"))
sys.path.insert(0, os.path.join(WS, "cc", "plm-evolve"))
sys.path.insert(0, os.path.join(WS, "030.repos_clone", "foldy", "backend"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import figure_lib as fl
from validate_gap2_dms import PRESETS
from plm_evolve.variants import read_wt_sequence
from folde.few_shot_models import RandomForestFewShotModel, TorchMLPFewShotModel


def load(dataset, model):
    ids, y, X = fl.load_aligned(dataset, [model], fl.DATASET_HASH[dataset])
    Xm = X[model]
    emb = pd.Series([Xm[i] for i in range(len(ids))], index=ids)
    act = pd.Series(y, index=ids)
    nat = pd.DataFrame(0.0, index=ids, columns=["nat"])
    wt = read_wt_sequence(PRESETS[dataset].wt_fasta)
    return ids, emb, act, nat, wt


def spear(factory, dataset, model, t=95, seeds=20):
    ids, emb, act, nat, wt = load(dataset, model)
    n = len(ids)
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.permutation(n)
        tr = [ids[i] for i in idx[:t]]
        te = [ids[i] for i in idx[t:]]
        m = factory(wt)
        m.fit(nat, emb, act.loc[tr])
        preds = m.predict(nat.loc[te], emb.loc[te])
        pred = pd.concat([p for p in preds], axis=1).mean(axis=1)
        vals.append(spearmanr(pred.values, act.loc[te].values).correlation)
    v = np.array(vals, float)
    return float(v.mean()), float(v.std(ddof=0) / np.sqrt(len(v)))


def rf(wt):
    return RandomForestFewShotModel(ensemble_size=1, wt_aa_seq=wt, random_state=1)


def bt(wt):
    return TorchMLPFewShotModel(ensemble_size=3, wt_aa_seq=wt, pretrain=False, random_state=0)


if __name__ == "__main__":
    import json
    E2, EC = fl.MEDIUM_PAIR
    print("== held-out Spearman, train=95, 20 seeds (identical embeddings/splits) ==", flush=True)
    print("RF = EVOLVEpro-core (== plm-evolve); BT = FolDE TorchMLP Bradley-Terry, no warm-start", flush=True)
    rows = {}
    for ds in fl.PROTEIN_ORDER:
        rows[ds] = {}
        for model, mlabel in [(E2, "ESM-2 150M"), (EC, "ESM-C 300M")]:
            rf_m, rf_se = spear(rf, ds, model)
            bt_m, bt_se = spear(bt, ds, model)
            rows[ds][mlabel] = {"RF": [rf_m, rf_se], "BT": [bt_m, bt_se], "delta": bt_m - rf_m}
            print(f"{ds:<11} {mlabel:<11} RF {rf_m:.3f}  BT {bt_m:.3f}  delta {bt_m - rf_m:+.3f}", flush=True)
    out = os.path.join(WS, "cc", "plm-evolve", "figures", "esm2_vs_esmc", "data", "folde_compare.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", out, flush=True)
