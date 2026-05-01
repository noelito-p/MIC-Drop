"""
02_train_classifier.py
----------------------
Aggregates cells -> wells, runs GroupKFold + GridSearchCV, compares
per-channel vs combined features.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
)

# >>> EDIT: paths
INPUT_CSV  = Path("/Users/u1935683/hackathon_data/output/cell_features.csv")
OUTPUT_DIR = Path("/Users/u1935683/hackathon_data/output")

ANTIBIOTIC_TO_MOA = {
    "Control":         "Control",
    "Triclosan":       "FattyAcid",
    "Novobiocin":      "DNA",
    "Naladixic_acid":  "DNA",
    "Rifampicin":      "RNA",
    "Chloramphenicol": "Protein",
    "Kanamycin":       "Protein",
    "Ampicillin":      "CellWall",
    "Mecillinam":      "RodShape",
    "A22":             "RodShape",
    "PMB":             "Membrane",
    "Gramacidin":      "Membrane",
    "Nisin":           "Membrane",
}

TARGET = "moa"  # or "antibiotic"

CHANNEL_FEATURE_GROUPS = {
    "DNA_only":          ["DNA"],
    "Permeability_only": ["Permeability"],
    "CellWall_only":     ["CellWall"],
    "Membrane_only":     ["Membrane"],
    "All_channels":      ["DNA", "Permeability", "CellWall", "Membrane"],
}

SHAPE_PREFIXES = [
    "area", "area_convex", "area_filled", "perimeter", "perimeter_crofton",
    "equivalent_diameter_area", "major_axis_length", "minor_axis_length",
    "eccentricity", "orientation", "solidity", "extent",
    "feret_diameter_max", "euler_number", "aspect_ratio", "circularity",
    "convexity_defect", "num_nucleoids", "nucleoid_area_total",
    "nucleoid_area_mean", "nucleoid_ratio", "nucleoid_intensity_mean",
    "nucleoid_eccentricity_mean", "nucleoid_elongation_mean",
    "nucleoid_centroid_offset",
]


def aggregate_to_well(df, feature_cols):
    grouped = df.groupby(["well_id", "antibiotic", "concentration", "condition"])
    rows = []
    for key, g in grouped:
        row = {"well_id": key[0], "antibiotic": key[1], "concentration": key[2],
               "condition": key[3], "n_cells": len(g)}
        if "membrane_compromised" in g.columns:
            row["frac_compromised"] = g["membrane_compromised"].mean()
        for col in feature_cols:
            if col not in g.columns: continue
            v = g[col].values
            row[f"{col}_mean"]   = np.mean(v)
            row[f"{col}_std"]    = np.std(v)
            row[f"{col}_median"] = np.median(v)
            row[f"{col}_p10"]    = np.percentile(v, 10)
            row[f"{col}_p90"]    = np.percentile(v, 90)
        rows.append(row)
    return pd.DataFrame(rows)


def select_columns(all_cols, channel_prefixes):
    sel = []
    for c in all_cols:
        if any(c.startswith(s + "_") or c == s for s in SHAPE_PREFIXES):
            sel.append(c); continue
        for p in channel_prefixes:
            if c.startswith(p + "_"):
                sel.append(c); break
    drop = {"well_id", "antibiotic", "concentration", "condition", "n_cells",
            "frac_compromised", "plate_position"}
    return [c for c in sel if c not in drop]


def train_and_eval(X, y, groups, name):
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")),
    ])
    grid = {
        "clf__n_estimators":     [200, 500],
        "clf__max_depth":        [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features":     ["sqrt", "log2"],
    }
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(X, y, groups=groups)

    y_pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv, groups=groups, n_jobs=-1)
    f1 = f1_score(y, y_pred, average="macro")

    print(f"\n=== {name} ===")
    print(f"Best params: {gs.best_params_}")
    print(f"CV F1: {f1:.3f}")
    print(classification_report(y, y_pred, zero_division=0))
    return gs.best_estimator_, y_pred, f1


def main():
    cells = pd.read_csv(INPUT_CSV)
    cells["moa"] = cells["antibiotic"].map(ANTIBIOTIC_TO_MOA)
    if cells["moa"].isna().any():
        missing = cells.loc[cells["moa"].isna(), "antibiotic"].unique()
        raise ValueError(f"No MoA mapping for: {missing}")

    drop = {"well_id", "antibiotic", "concentration", "condition", "label",
            "moa", "centroid-0", "centroid-1", "membrane_compromised",
            "bbox-0", "bbox-1", "bbox-2", "bbox-3", "plate_position"}
    feat_cols = [c for c in cells.columns
                 if c not in drop and pd.api.types.is_numeric_dtype(cells[c])]

    well_df = aggregate_to_well(cells, feat_cols)
    well_df["moa"] = well_df["antibiotic"].map(ANTIBIOTIC_TO_MOA)
    print(f"\n{len(well_df)} wells; class counts:")
    print(well_df[TARGET].value_counts())
    well_df.to_csv(OUTPUT_DIR / "well_features.csv", index=False)

    results = {}
    for name, prefixes in CHANNEL_FEATURE_GROUPS.items():
        cols = [c for c in select_columns(well_df.columns, prefixes) if c in well_df.columns]
        if not cols: continue
        X = well_df[cols].fillna(0).values
        y = well_df[TARGET].values
        groups = well_df["well_id"].values
        model, y_pred, f1 = train_and_eval(X, y, groups, name)
        results[name] = {"model": model, "y_pred": y_pred, "y_true": y,
                         "f1": f1, "features": cols}
        joblib.dump(model, OUTPUT_DIR / f"model_{name}.joblib")

    # Comparison plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys()); scores = [results[n]["f1"] for n in names]
    sns.barplot(x=names, y=scores, ax=ax)
    ax.set_ylabel("Macro F1 (Group-CV)"); ax.set_ylim(0, 1)
    ax.set_title("Per-channel vs combined performance")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "channel_comparison.png", dpi=150); plt.close()

    # Confusion matrix
    best = results.get("All_channels", results[names[0]])
    labels = sorted(np.unique(best["y_true"]))
    cm = confusion_matrix(best["y_true"], best["y_pred"], labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_combined.png", dpi=150); plt.close()

    # Feature importances
    rf = best["model"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=best["features"]) \
            .sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.iloc[::-1].plot.barh(ax=ax)
    ax.set_title("Top 20 features (RF, all channels)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importances.png", dpi=150); plt.close()

    pd.DataFrame({"feature_set": names, "macro_f1": scores}).to_csv(
        OUTPUT_DIR / "results_summary.csv", index=False)
    print("\nDone:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
