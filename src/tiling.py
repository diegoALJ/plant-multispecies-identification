"""Image tiling utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def apply_border_crop(img: Image.Image, border_crop_pct: float = 0.0) -> Image.Image:
    """Crop a small border from the image to reduce frame artifacts."""
    if border_crop_pct <= 0:
        return img

    w, h = img.size
    dx = int(w * border_crop_pct)
    dy = int(h * border_crop_pct)
    box = (dx, dy, w - dx, h - dy)

    if box[2] <= box[0] or box[3] <= box[1]:
        return img

    return img.crop(box)


def generate_grid_tiles(
    img: Image.Image,
    grid_size: int = 5,
    include_full_image: bool = True,
) -> List[Dict]:
    """Generate full-image tile plus fixed grid crops."""
    tiles = []
    w, h = img.size
    tile_id = 0

    if include_full_image:
        tiles.append({
            "tile_id": tile_id,
            "tile_type": "full",
            "row": -1,
            "col": -1,
            "box": (0, 0, w, h),
            "image": img.copy(),
        })
        tile_id += 1

    x_edges = np.linspace(0, w, grid_size + 1).astype(int)
    y_edges = np.linspace(0, h, grid_size + 1).astype(int)

    for row in range(grid_size):
        for col in range(grid_size):
            box: Tuple[int, int, int, int] = (
                int(x_edges[col]),
                int(y_edges[row]),
                int(x_edges[col + 1]),
                int(y_edges[row + 1]),
            )
            tiles.append({
                "tile_id": tile_id,
                "tile_type": "grid",
                "row": row,
                "col": col,
                "box": box,
                "image": img.crop(box),
            })
            tile_id += 1

    return tiles


def vegetation_score_rgb(img: Image.Image) -> float:
    """Simple green-dominance score used as a rough non-plant risk flag."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_dominance = g - 0.5 * (r + b)
    return float(np.mean(green_dominance > 10))
