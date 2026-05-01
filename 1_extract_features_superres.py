"""
01_extract_features_superres.py
-------------------------------
Same as 01_extract_features.py but:
  - Reads super-res images (4096x4096) from super_res_v2.zarr
  - Upscales widefield masks (2048x2048) to 4096x4096 by nearest-neighbor
  - Saves to a separate output dir so it doesn't overwrite widefield results

Run: python 01_extract_features_superres.py
"""

import yaml
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom

from analyze_cells_expanded import analyze_cells_expanded

ZARR_PATH       = Path("/Users/u1935683/hackathon_data/super_res_v2.zarr")
MASK_ZARR_PATH  = Path("/Users/u1935683/hackathon_data/masks_v2.zarr")
YAML_PATH       = Path("/Users/u1935683/hackathon_data/Antibiotic_positions.yaml")
OUTPUT_DIR      = Path("/Users/u1935683/hackathon_data/output_superres")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_NAMES = {0: "DNA", 1: "Permeability", 2: "CellWall", 3: "Membrane"}

# Cell size filtering — IMPORTANT: super-res cells are 4× larger in pixels
# than widefield because each linear dimension is 2× bigger.
# Widefield used MIN=30, MAX=8000. Multiply by 4 for super-res.
MIN_CELL_AREA = 120     # 30 * 4
MAX_CELL_AREA = 32000   # 8000 * 4

PERMEABILITY_THRESHOLD_PERCENTILE = 95
CONTROL_NAME = "Control"


def load_well_image(zarr_root, well_index):
    arr = np.asarray(zarr_root[f"p{well_index}"]["0"])  # (4, 4096, 4096)
    if arr.ndim != 3 or arr.shape[0] != 4:
        raise ValueError(f"Unexpected image shape: {arr.shape}")
    return {CHANNEL_NAMES[i]: arr[i] for i in range(4)}


def load_well_mask_upscaled(mask_array, well_index, target_shape):
    """Load mask and nearest-neighbor upscale to target_shape."""
    mask = np.asarray(mask_array[well_index]).astype(np.int32)
    if mask.shape == target_shape:
        return mask
    factor_y = target_shape[0] / mask.shape[0]
    factor_x = target_shape[1] / mask.shape[1]
    # Nearest-neighbor — preserves integer cell labels
    upscaled = zoom(mask, (factor_y, factor_x), order=0).astype(np.int32)
    if upscaled.shape != target_shape:
        # Crop or pad to exact target
        result = np.zeros(target_shape, dtype=np.int32)
        h = min(upscaled.shape[0], target_shape[0])
        w = min(upscaled.shape[1], target_shape[1])
        result[:h, :w] = upscaled[:h, :w]
        upscaled = result
    return upscaled


def main():
    with open(YAML_PATH) as f:
        meta = yaml.safe_load(f)
    names = meta["names"]
    positions = meta["positions"]
    print(f"Loaded {len(names)} conditions")

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
            channels = load_well_image(data_root, i)
            target_shape = next(iter(channels.values())).shape
            cell_mask = load_well_mask_upscaled(mask_arr, i, target_shape)
        except Exception as e:
            print(f"  Skip {name} (p{i}): load error — {e}")
            skipped.append((name, i, str(e)))
            continue

        if cell_mask.max() == 0:
            print(f"  Skip {name} (p{i}): empty mask")
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

    if (full["antibiotic"] == "Control").any():
        ctrl = full[full["antibiotic"] == "Control"]
        if "Permeability_mean" in ctrl.columns and len(ctrl) > 0:
            thresh = np.percentile(ctrl["Permeability_mean"],
                                   PERMEABILITY_THRESHOLD_PERCENTILE)
            full["membrane_compromised"] = (full["Permeability_mean"] > thresh).astype(int)

    out = OUTPUT_DIR / "cell_features.csv"
    full.to_csv(out, index=False)
    print(f"\nSaved {len(full)} cells -> {out}")

    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for n, i, e in skipped:
            print(f"  {n} (p{i}): {e}")


if __name__ == "__main__":
    main()
