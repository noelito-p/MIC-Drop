"""
06_train_with_deep.py
---------------------
Trains classifiers comparing four feature sets:
  1. Handcrafted (your existing morphology features)
  2. ResNet18 deep features
  3. DINOv2 deep features
  4. Handcrafted + ResNet18 + DINOv2 (concat)

Aggregates to well-level, GroupKFold + GridSearchCV.
Predicts both MoA and antibiotic.

Run: python 06_train_with_deep.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
)

# >>> EDIT
HANDCRAFTED_CSV = Path("/Users/u1935683/hackathon_data/output/cell_features.csv")
RESNET_CSV      = Path("/Users/u1935683/hackathon_data/output/deep_features_resnet18.csv")
DINOV2_CSV      = Path("/Users/u1935683/hackathon_data/output/deep_features_dinov2.csv")
OUTPUT_DIR      = Path("/Users/u1935683/hackathon_data/output")

# >>> EDIT: target — "moa" or "antibiotic"
TARGET = "antibiotic"

DROP_CONTROL = True

ANTIBIOTIC_TO_MOA = {
    "Control": "Control", "Triclosan": "FattyAcid", "Novobiocin": "DNA",
    "Naladixic_acid": "DNA", "Rifampicin": "RNA", "Chloramphenicol": "Protein",
    "Kanamycin": "Protein", "Ampicillin": "CellWall", "Mecillinam": "RodShape",
    "A22": "RodShape", "PMB": "Membrane", "Gramacidin": "Membrane",
    "Nisin": "Membrane",
}


def aggregate_to_well(df, feat_cols):
    """Mean across cells per well (deep features) — std/percentiles add too much dim."""
    rows = []
    for key, g in df.groupby(["well_id", "antibiotic", "concentration", "condition"]):
        row = {"well_id": key[0], "antibiotic": key[1],
               "concentration": key[2], "condition": key[3], "n_cells": len(g)}
        for c in feat_cols:
            if c in g.columns:
                row[f"{c}_mean"] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_handcrafted(df, feat_cols):
    """Handcrafted features get full aggregation (mean, std, percentiles)."""
    rows = []
    for key, g in df.groupby(["well_id", "antibiotic", "concentration", "condition"]):
        row = {"well_id": key[0], "antibiotic": key[1],
               "concentration": key[2], "condition": key[3], "n_cells": len(g)}
        for c in feat_cols:
            if c not in g.columns: continue
            v = g[c].values
            row[f"{c}_mean"]   = np.mean(v)
            row[f"{c}_std"]    = np.std(v)
            row[f"{c}_p10"]    = np.percentile(v, 10)
            row[f"{c}_p90"]    = np.percentile(v, 90)
        rows.append(row)
    return pd.DataFrame(rows)


def get_feature_columns(df):
    drop = {"well_id", "antibiotic", "concentration", "condition", "label",
            "moa", "n_cells", "centroid-0", "centroid-1", "membrane_compromised",
            "bbox-0", "bbox-1", "bbox-2", "bbox-3", "plate_position"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def train_eval(X, y, groups, name):
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=30)),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")),
    ])
    grid = {
        "select__k":             [20, 50, 100],
        "clf__n_estimators":     [200, 500],
        "clf__min_samples_leaf": [1, 2],
    }
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
    gs.fit(X, y, groups=groups)

    y_pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv, groups=groups, n_jobs=-1)
    f1 = f1_score(y, y_pred, average="macro")
    print(f"\n=== {name} ===")
    print(f"Features: {X.shape[1]}, Best k: {gs.best_params_['select__k']}")
    print(f"Macro F1: {f1:.3f}")
    return gs.best_estimator_, y_pred, f1


def main():
    # ---- Load handcrafted ----
    hc = pd.read_csv(HANDCRAFTED_CSV)
    hc["moa"] = hc["antibiotic"].map(ANTIBIOTIC_TO_MOA)
    if DROP_CONTROL:
        hc = hc[hc["antibiotic"] != "Control"]
    hc_feat_cols = get_feature_columns(hc)
    well_hc = aggregate_handcrafted(hc, hc_feat_cols)
    well_hc["moa"] = well_hc["antibiotic"].map(ANTIBIOTIC_TO_MOA)

    # ---- Load deep features ----
    feature_sets = {"Handcrafted": well_hc}

    if RESNET_CSV.exists():
        rn = pd.read_csv(RESNET_CSV)
        if DROP_CONTROL:
            rn = rn[rn["antibiotic"] != "Control"]
        rn_cols = [c for c in rn.columns if c.startswith("resnet18_")]
        well_rn = aggregate_to_well(rn, rn_cols)
        well_rn["moa"] = well_rn["antibiotic"].map(ANTIBIOTIC_TO_MOA)
        feature_sets["ResNet18"] = well_rn

    if DINOV2_CSV.exists():
        dn = pd.read_csv(DINOV2_CSV)
        if DROP_CONTROL:
            dn = dn[dn["antibiotic"] != "Control"]
        dn_cols = [c for c in dn.columns if c.startswith("dinov2_")]
        well_dn = aggregate_to_well(dn, dn_cols)
        well_dn["moa"] = well_dn["antibiotic"].map(ANTIBIOTIC_TO_MOA)
        feature_sets["DINOv2"] = well_dn

    # ---- Combined: merge on well_id ----
    if len(feature_sets) > 1:
        merged = well_hc[["well_id", "antibiotic", "concentration", "condition", "moa"]].copy()
        for name, fset in feature_sets.items():
            feat_cols = [c for c in fset.columns
                         if c not in ["well_id", "antibiotic", "concentration",
                                      "condition", "moa", "n_cells"]]
            merged = merged.merge(fset[["well_id"] + feat_cols], on="well_id", how="inner")
        feature_sets["Combined"] = merged

    # ---- Train each ----
    results = {}
    for name, df in feature_sets.items():
        meta_cols = ["well_id", "antibiotic", "concentration", "condition", "moa", "n_cells"]
        feat_cols = [c for c in df.columns if c not in meta_cols
                     and pd.api.types.is_numeric_dtype(df[c])]
        if not feat_cols:
            continue
        X = df[feat_cols].fillna(0).values
        y = df[TARGET].values
        groups = df["well_id"].values
        model, y_pred, f1 = train_eval(X, y, groups, name)
        results[name] = {"model": model, "y_pred": y_pred, "y_true": y, "f1": f1,
                         "labels": sorted(np.unique(y))}

    # ---- Comparison plot ----
    names  = list(results.keys())
    scores = [results[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, scores, color=["steelblue", "coral", "seagreen", "orchid"][:len(names)])
    ax.set_ylabel(f"Macro F1 ({TARGET} prediction)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Feature comparison: handcrafted vs deep — {TARGET}")
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width()/2, s + 0.02, f"{s:.2f}",
                ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"feature_comparison_{TARGET}.png", dpi=150)
    plt.close()

    # ---- Confusion matrix for best model ----
    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    cm = confusion_matrix(best["y_true"], best["y_pred"], labels=best["labels"])
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=best["labels"]).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"Best: {best_name} on {TARGET} (F1={best['f1']:.2f})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_best_{TARGET}.png", dpi=150)
    plt.close()

    # Save summary
    pd.DataFrame({"feature_set": names, "macro_f1": scores}).to_csv(
        OUTPUT_DIR / f"deep_results_{TARGET}.csv", index=False)
    print(f"\nDone. Best: {best_name} (F1={best['f1']:.3f})")
    print(f"Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
