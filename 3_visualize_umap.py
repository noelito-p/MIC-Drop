"""
03_visualize_umap.py — fixed for color consistency with 3D UMAP
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap

INPUT_CSV  = Path("/Users/u1935683/hackathon_data/output_superres/cell_features.csv")
OUTPUT_DIR = Path("/Users/u1935683/hackathon_data/output_superres")

ANTIBIOTIC_TO_MOA = {
    "Control": "Control", "Triclosan": "FattyAcid", "Novobiocin": "DNA",
    "Naladixic_acid": "DNA", "Rifampicin": "RNA", "Chloramphenicol": "Protein",
    "Kanamycin": "Protein", "Ampicillin": "CellWall", "Mecillinam": "RodShape",
    "A22": "RodShape", "PMB": "Membrane", "Gramacidin": "Membrane",
    "Nisin": "Membrane",
}

ANTIBIOTIC_ORDER = sorted(["A22", "Ampicillin", "Chloramphenicol", "Control",
    "Gramacidin", "Kanamycin", "Mecillinam", "Naladixic_acid", "Nisin",
    "Novobiocin", "PMB", "Rifampicin", "Triclosan"])
MOA_ORDER = sorted(["CellWall", "Control", "DNA", "FattyAcid", "Membrane",
                    "Protein", "RNA", "RodShape"])

def build_color_dict(order, palette_name):
    cmap = plt.get_cmap(palette_name)
    return {v: cmap(i) for i, v in enumerate(order)}

ANTIBIOTIC_COLORS = build_color_dict(ANTIBIOTIC_ORDER, "tab20")
MOA_COLORS        = build_color_dict(MOA_ORDER, "tab10")

MAX_CELLS_PER_WELL = 200

def main():
    df = pd.read_csv(INPUT_CSV)
    df["moa"] = df["antibiotic"].map(ANTIBIOTIC_TO_MOA)

    parts = []
    for w in df["well_id"].unique():
        sub = df[df["well_id"] == w]
        n = min(len(sub), MAX_CELLS_PER_WELL)
        parts.append(sub.sample(n=n, random_state=42))
    df = pd.concat(parts, ignore_index=True)

    drop = {"well_id", "antibiotic", "concentration", "condition", "label",
            "moa", "centroid-0", "centroid-1", "membrane_compromised",
            "bbox-0", "bbox-1", "bbox-2", "bbox-3", "plate_position"}
    feat_cols = [c for c in df.columns
                 if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    X = StandardScaler().fit_transform(df[feat_cols].fillna(0).values)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(X)

    plot_df = pd.DataFrame({"UMAP1": emb[:, 0], "UMAP2": emb[:, 1],
                            "Antibiotic": df["antibiotic"].values,
                            "MoA": df["moa"].values})

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="UMAP1", y="UMAP2", hue="Antibiotic",
                    palette=ANTIBIOTIC_COLORS,
                    hue_order=[v for v in ANTIBIOTIC_ORDER
                               if v in plot_df["Antibiotic"].values],
                    s=5, alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=3, fontsize=8)
    plt.title("Single-cell morphology UMAP — by antibiotic")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap_by_antibiotic.png", dpi=150); plt.close()

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=plot_df, x="UMAP1", y="UMAP2", hue="MoA",
                    palette=MOA_COLORS,
                    hue_order=[v for v in MOA_ORDER if v in plot_df["MoA"].values],
                    s=5, alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=3)
    plt.title("Single-cell morphology UMAP — by MoA")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap_by_moa.png", dpi=150); plt.close()
    print("Saved 2D UMAPs with consistent colors.")


if __name__ == "__main__":
    main()
