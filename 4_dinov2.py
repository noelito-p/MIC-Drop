"""
05_deep_features.py
-------------------
Extracts deep features per cell using pretrained models (ResNet18 and DINOv2).
For each segmented cell:
  1. Crop a fixed-size patch
  2. Pass through pretrained CNN/ViT
  3. Save the embedding

Saves: deep_features_resnet18.csv and deep_features_dinov2.csv
"""

import numpy as np
import pandas as pd
import zarr
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# >>> EDIT: paths (point at the v2-converted zarrs)
# =============================================================================
ZARR_PATH      = Path("/Users/u1935683/hackathon_data/Widefield_v2.zarr")
MASK_ZARR_PATH = Path("/Users/u1935683/hackathon_data/masks_v2.zarr")
CSV_PATH       = Path("/Users/u1935683/hackathon_data/output/cell_features.csv")
OUTPUT_DIR     = Path("/Users/u1935683/hackathon_data/output")

# >>> EDIT: which models to run. Run them one at a time for safety.
RUN_RESNET = True
RUN_DINOV2 = True    # set True after ResNet succeeds

PATCH_SIZE = 96
MAX_CELLS_PER_WELL = 100   # ~3700 cells total, manageable on CPU


def get_device():
    # Force CPU for stability — MPS has been segfaulting
    return torch.device("cpu")
    # If you want to try MPS later:
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    # return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")


# =============================================================================
# Cropping & preprocessing
# =============================================================================
def crop_cell(image_3channels, centroid, size=PATCH_SIZE):
    """Crop a (C, size, size) patch around centroid, padding edges."""
    cy, cx = int(centroid[0]), int(centroid[1])
    half = size // 2
    _, H, W = image_3channels.shape
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)
    patch = image_3channels[:, y0:y1, x0:x1]
    out = np.zeros((image_3channels.shape[0], size, size), dtype=patch.dtype)
    out[:, :patch.shape[1], :patch.shape[2]] = patch
    return out


def to_rgb(patch_4ch):
    """Map 4-channel microscopy to 3-channel RGB-like:
    R = CellWall (ch2), G = DNA (ch0), B = Membrane (ch3)."""
    rgb = np.stack([patch_4ch[2], patch_4ch[0], patch_4ch[3]], axis=0)
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        c = rgb[i]
        lo, hi = np.percentile(c, [1, 99])
        if hi > lo:
            out[i] = np.clip((c - lo) / (hi - lo), 0, 1)
    return out


def build_resnet18():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval().to(DEVICE)
    return m

def build_dinov2_small():
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    m.eval().to(DEVICE)
    return m

imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def resnet_transform(rgb_np):
    t = torch.from_numpy(rgb_np).float()
    t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(224, 224),
                                         mode="bilinear", align_corners=False)[0]
    return imagenet_norm(t)

def dinov2_transform(rgb_np):
    t = torch.from_numpy(rgb_np).float()
    t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(224, 224),
                                         mode="bilinear", align_corners=False)[0]
    return imagenet_norm(t)


# =============================================================================
# Main extraction loop
# =============================================================================
@torch.no_grad()
def extract_with_model(model, transform_fn, cells_df, data_root, mask_arr, model_name):
    print(f"\n--- {model_name} ---")

    # ---- Subsample cells per well, robustly ----
    if MAX_CELLS_PER_WELL is not None:
        sampled = []
        for well_id in cells_df["well_id"].unique():
            sub = cells_df[cells_df["well_id"] == well_id]
            n = min(len(sub), MAX_CELLS_PER_WELL)
            sampled.append(sub.sample(n=n, random_state=42))
        cells_df = pd.concat(sampled, ignore_index=True)
    print(f"  Working with {len(cells_df)} cells")

    embeddings = []
    metadata = []

    # Iterate over unique well IDs explicitly (avoids groupby quirks)
    unique_wells = cells_df["well_id"].unique()
    BATCH_SIZE = 16   # smaller for CPU safety

    for well_id in tqdm(unique_wells, desc=model_name):
        group = cells_df[cells_df["well_id"] == well_id]
        well_idx = int(well_id[1:])

        # Load this well's image once
        img = np.asarray(data_root[well_id]["0"]).astype(np.float32)  # (4, H, W)

        rows = group.to_dict("records")
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            patches = []
            for r in batch:
                p4 = crop_cell(img, (r["centroid-0"], r["centroid-1"]))
                rgb = to_rgb(p4)
                patches.append(transform_fn(rgb))
            tensor = torch.stack(patches).to(DEVICE)
            feats = model(tensor).cpu().numpy()
            embeddings.append(feats)
            for r in batch:
                metadata.append({
                    "well_id":       r["well_id"],
                    "label":         r["label"],
                    "antibiotic":    r["antibiotic"],
                    "concentration": r["concentration"],
                    "condition":     r["condition"],
                })

    embeddings = np.vstack(embeddings)
    print(f"  Extracted {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    df_meta = pd.DataFrame(metadata)
    df_feat = pd.DataFrame(embeddings,
                           columns=[f"{model_name}_{i}" for i in range(embeddings.shape[1])])
    df = pd.concat([df_meta, df_feat], axis=1)
    out = OUTPUT_DIR / f"deep_features_{model_name}.csv"
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    return df


def main():
    cells = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(cells)} cells from {cells['well_id'].nunique()} wells")

    if "centroid-0" not in cells.columns:
        raise ValueError("CSV missing 'centroid-0' column. Re-run script 01.")
    if "well_id" not in cells.columns:
        raise ValueError("CSV missing 'well_id' column.")

    data_root = zarr.open(str(ZARR_PATH), mode="r")
    mask_arr  = zarr.open(str(MASK_ZARR_PATH), mode="r")

    if RUN_RESNET:
        model = build_resnet18()
        extract_with_model(model, resnet_transform, cells, data_root, mask_arr,
                           "resnet18")
        del model

    if RUN_DINOV2:
        model = build_dinov2_small()
        extract_with_model(model, dinov2_transform, cells, data_root, mask_arr,
                           "dinov2")

    print("\nDone. Use 06_train_with_deep.py to train classifiers on these features.")


if __name__ == "__main__":
    main()