"""
01_extract_features.py
----------------------
Reads Widefield.zarr (image groups p0...p127) and masks.zarr (single 128xYxX array),
extracts features for each well listed in the YAML, saves cell_features.csv.

Run: python 01_extract_features.py
"""

import yaml
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from tqdm import tqdm

from analyze_cells_expanded import analyze_cells_expanded

# =============================================================================
# >>> EDIT IF NEEDED — paths
# =============================================================================
ZARR_PATH       = Path("/Users/u1935683/hackathon_data/Widefield.zarr")
MASK_ZARR_PATH  = Path("/Users/u1935683/hackathon_data/masks.zarr")
YAML_PATH       = Path("/Users/u1935683/hackathon_data/Antibiotic_positions.yaml")
OUTPUT_DIR      = Path("/Users/u1935683/hackathon_data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Channel order in the (4, Y, X) image array
CHANNEL_NAMES = {
    0: "DNA",
    1: "Permeability",
    2: "CellWall",
    3: "Membrane",
}

# Cell area filtering (pixels). 2048x2048 images, E. coli ~1-3 um wide
MIN_CELL_AREA = 30
MAX_CELL_AREA = 8000

# Permeability threshold percentile (relative to control well)
PERMEABILITY_THRESHOLD_PERCENTILE = 95

CONTROL_NAME = "Control"


def load_well_image(zarr_root, well_index):
    """Load image for well index N: returns dict of {channel_name: 2D array}."""
    arr = np.asarray(zarr_root[f"p{well_index}"]["0"])
    # arr shape is (4, 2048, 2048)
    if arr.ndim != 3 or arr.shape[0] != 4:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    return {CHANNEL_NAMES[i]: arr[i] for i in range(4)}


def load_well_mask(mask_array, well_index):
    """Load mask for well index N from the stacked masks array."""
    return np.asarray(mask_array[well_index]).astype(np.int32)


def main():
    with open(YAML_PATH) as f:
        meta = yaml.safe_load(f)
    names = meta["names"]
    positions = meta["positions"]

    if len(names) != len(positions):
        print(f"WARNING: YAML has {len(names)} names but {len(positions)} positions")

    print(f"Loaded {len(names)} conditions from YAML")

    # Open zarr
    data_root = zarr.open(str(ZARR_PATH), mode="r")
    mask_arr  = zarr.open(str(MASK_ZARR_PATH), mode="r")
    print(f"Image zarr: {len(list(data_root.keys()))} wells")
    print(f"Mask array: shape {mask_arr.shape}")

    n_to_process = min(len(names), mask_arr.shape[0])

    all_features = []
    skipped = []
    for i in tqdm(range(n_to_process), desc="Wells"):
        name = names[i]
        position = positions[i] if i < len(positions) else "?"

        if name.lower() == "blank":
            continue

        if name == CONTROL_NAME:
            antibiotic, concentration = "Control", "Control"
        else:
            parts = name.rsplit("_", 1)
            antibiotic = parts[0]
            concentration = parts[1] if len(parts) > 1 else "?"

        try:
            channels  = load_well_image(data_root, i)
            cell_mask = load_well_mask(mask_arr, i)
        except Exception as e:
            print(f"  Skip {name} (p{i}): load error — {e}")
            skipped.append((name, i, str(e)))
            continue

        if cell_mask.max() == 0:
            print(f"  Skip {name} (p{i}): mask is empty")
            continue

        try:
            df = analyze_cells_expanded(cell_mask, channels)
        except Exception as e:
            print(f"  Skip {name} (p{i}): feature error — {e}")
            skipped.append((name, i, str(e)))
            continue

        df = df[(df["area"] >= MIN_CELL_AREA) & (df["area"] <= MAX_CELL_AREA)]
        df["well_id"]       = f"p{i}"
        df["plate_position"] = position
        df["antibiotic"]    = antibiotic
        df["concentration"] = concentration
        df["condition"]     = name
        all_features.append(df)
        print(f"  {name} (p{i}): {len(df)} cells")

    if not all_features:
        raise RuntimeError("No wells processed successfully")

    full = pd.concat(all_features, ignore_index=True)

    # Add membrane-compromised flag using control percentile
    if (full["antibiotic"] == "Control").any():
        ctrl = full[full["antibiotic"] == "Control"]
        if "Permeability_mean" in ctrl.columns and len(ctrl) > 0:
            thresh = np.percentile(ctrl["Permeability_mean"],
                                   PERMEABILITY_THRESHOLD_PERCENTILE)
            full["membrane_compromised"] = (full["Permeability_mean"] > thresh).astype(int)
            print(f"\nPermeability threshold from control = {thresh:.2f}")
            print(f"% compromised cells overall: {full['membrane_compromised'].mean()*100:.1f}%")

    out = OUTPUT_DIR / "cell_features.csv"
    full.to_csv(out, index=False)
    print(f"\nSaved {len(full)} cells from {full['well_id'].nunique()} wells -> {out}")

    if skipped:
        print(f"\n{len(skipped)} wells skipped:")
        for n, i, e in skipped:
            print(f"  {n} (p{i}): {e}")


if __name__ == "__main__":
    main()