"""
08_umap_3d.py — fixed for color consistency with 2D UMAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
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

# Class order — must match the order seaborn assigns colors when the column is sorted.
# Seaborn assigns colors based on the SORTED unique values from the data.
# Both 2D and 3D will now use the same explicit dictionaries below.
ANTIBIOTIC_ORDER = sorted([
    "A22", "Ampicillin", "Chloramphenicol", "Control", "Gramacidin",
    "Kanamycin", "Mecillinam", "Naladixic_acid", "Nisin", "Novobiocin",
    "PMB", "Rifampicin", "Triclosan",
])
MOA_ORDER = sorted([
    "CellWall", "Control", "DNA", "FattyAcid", "Membrane",
    "Protein", "RNA", "RodShape",
])

def build_color_dict(order, palette_name):
    cmap = plt.get_cmap(palette_name)
    return {v: cmap(i) for i, v in enumerate(order)}

ANTIBIOTIC_COLORS = build_color_dict(ANTIBIOTIC_ORDER, "tab20")
MOA_COLORS        = build_color_dict(MOA_ORDER, "tab10")

DROP_CONTROL = False
MAX_CELLS_PER_WELL = 200


def main():
    df = pd.read_csv(INPUT_CSV)
    if DROP_CONTROL:
        df = df[df["antibiotic"] != "Control"]
    df["moa"] = df["antibiotic"].map(ANTIBIOTIC_TO_MOA)

    parts = []
    for w in df["well_id"].unique():
        sub = df[df["well_id"] == w]
        n = min(len(sub), MAX_CELLS_PER_WELL)
        parts.append(sub.sample(n=n, random_state=42))
    df = pd.concat(parts, ignore_index=True)
    print(f"Cells in UMAP: {len(df)}")

    drop = {"well_id", "antibiotic", "concentration", "condition", "label",
            "moa", "centroid-0", "centroid-1", "membrane_compromised",
            "bbox-0", "bbox-1", "bbox-2", "bbox-3", "plate_position"}
    feat_cols = [c for c in df.columns
                 if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    X = StandardScaler().fit_transform(df[feat_cols].fillna(0).values)
    print("Running 3D UMAP...")
    emb = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1,
                    random_state=42).fit_transform(X)
    df["UMAP1"] = emb[:, 0]; df["UMAP2"] = emb[:, 1]; df["UMAP3"] = emb[:, 2]

    angles = [(20, 30), (20, 120), (20, 210), (80, 0)]
    angle_titles = ["View 1", "View 2", "View 3", "Top-down"]

    for color_by, color_dict in [("antibiotic", ANTIBIOTIC_COLORS),
                                  ("moa", MOA_COLORS)]:
        unique = [v for v in color_dict.keys() if v in df[color_by].unique()]

        fig = plt.figure(figsize=(14, 12))
        for i, ((elev, azim), title) in enumerate(zip(angles, angle_titles)):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            for v in unique:
                sub = df[df[color_by] == v]
                ax.scatter(sub["UMAP1"], sub["UMAP2"], sub["UMAP3"],
                           c=[color_dict[v]], s=4, alpha=0.6, label=v)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_zlabel("UMAP3")
            ax.set_title(title, fontsize=11)
            if i == 0:
                ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left",
                          fontsize=8, markerscale=2.5)

        plt.suptitle(f"3D UMAP — colored by {color_by}", fontsize=13, y=1.0)
        plt.tight_layout()
        out = OUTPUT_DIR / f"umap_3d_by_{color_by}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    try:
        import plotly.express as px
        for color_by, color_dict in [("antibiotic", ANTIBIOTIC_COLORS),
                                       ("moa", MOA_COLORS)]:
            color_hex = {k: f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
                         for k, c in color_dict.items()}
            fig = px.scatter_3d(df, x="UMAP1", y="UMAP2", z="UMAP3",
                                color=color_by,
                                color_discrete_map=color_hex,
                                hover_data=["antibiotic", "concentration", "well_id"],
                                opacity=0.7,
                                title=f"3D UMAP — by {color_by}")
            fig.update_traces(marker=dict(size=2.5))
            out = OUTPUT_DIR / f"umap_3d_by_{color_by}.html"
            fig.write_html(str(out))
            print(f"Saved interactive: {out}")
    except ImportError:
        print("Install plotly for interactive: pip install plotly")

    df[["well_id", "label", "antibiotic", "concentration", "condition", "moa",
        "UMAP1", "UMAP2", "UMAP3"]].to_csv(
        OUTPUT_DIR / "umap_3d_coords.csv", index=False)


if __name__ == "__main__":
    main()
