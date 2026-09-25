"""Data loading helpers for PlantCLEF2026."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_EXTS = [
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP", ".WEBP",
]


def load_test_metadata(path: str | Path) -> pd.DataFrame:
    """Load semicolon-separated PlantCLEF test metadata."""
    df = pd.read_csv(path, sep=";", low_memory=False)
    df["quadrat_id"] = df["quadrat_id"].astype(str)
    return df


def load_species_ids(path: str | Path) -> pd.DataFrame:
    """Load official species IDs."""
    df = pd.read_csv(path, low_memory=False)
    df["species_id"] = df["species_id"].astype(str)
    return df


def find_images(root: str | Path, image_exts: Iterable[str] = IMAGE_EXTS) -> pd.DataFrame:
    """Find image files recursively and return a dataframe of paths."""
    root = Path(root)
    paths = []
    for ext in image_exts:
        paths.extend(root.rglob(f"*{ext}"))
    paths = sorted(set(paths))

    return pd.DataFrame({
        "image_path": [str(p) for p in paths],
        "filename": [p.name for p in paths],
        "stem": [p.stem for p in paths],
        "suffix": [p.suffix for p in paths],
    })


def attach_image_paths(test_meta: pd.DataFrame, image_df: pd.DataFrame) -> pd.DataFrame:
    """Attach image paths to test metadata using quadrat_id == image stem."""
    stem_to_path = dict(zip(image_df["stem"], image_df["image_path"]))
    out = test_meta.copy()
    out["image_path"] = out["quadrat_id"].map(stem_to_path)
    return out


def parse_quadrat_groups(test_meta: pd.DataFrame) -> pd.DataFrame:
    """Create simple group columns from quadrat_id."""
    out = test_meta.copy()
    split_ids = out["quadrat_id"].astype(str).str.split("-")
    out["id_prefix_1"] = split_ids.str[:1].str.join("-")
    out["id_prefix_2"] = split_ids.str[:2].str.join("-")
    out["id_prefix_3"] = split_ids.str[:3].str.join("-")
    out["id_without_date"] = out["quadrat_id"].str.replace(r"-\d{8}$", "", regex=True)
    out["transect_guess"] = out["id_without_date"].str.replace(r"\d+$", "", regex=True)
    return out
