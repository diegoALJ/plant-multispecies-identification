"""Submission and post-processing utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def format_species_list(species_list: Iterable[str]) -> str:
    """Format species IDs for Kaggle submission."""
    return "[" + ", ".join([str(x) for x in species_list]) + "]"


def prediction_length_from_string(pred_str: str) -> int:
    """Count species IDs inside a formatted prediction string."""
    pred_str = str(pred_str).strip().strip("[]")
    if pred_str == "":
        return 0
    return len([x for x in pred_str.split(",") if x.strip()])


def select_fixed_topk(
    scores_row: np.ndarray,
    species_values: np.ndarray,
    topk: int,
    threshold: float,
) -> List[str]:
    """Select species using fixed top-k and score threshold."""
    sorted_idx = np.argsort(scores_row)[::-1]
    selected = []

    for idx in sorted_idx:
        if len(selected) >= topk:
            break
        if scores_row[idx] >= threshold:
            selected.append(species_values[idx])

    if not selected:
        selected = [species_values[sorted_idx[0]]]

    return selected


def create_submission(
    scores: np.ndarray,
    quadrat_values: np.ndarray,
    species_values: np.ndarray,
    topk: int,
    threshold: float,
) -> pd.DataFrame:
    """Create a Kaggle submission dataframe from image-level scores."""
    rows = []
    for i in range(scores.shape[0]):
        selected = select_fixed_topk(scores[i], species_values, topk, threshold)
        rows.append({
            "quadrat_id": quadrat_values[i],
            "species_ids": format_species_list(selected),
        })
    return pd.DataFrame(rows)


def validate_submission(submission: pd.DataFrame, expected_n_rows: int) -> bool:
    """Basic format checks for a PlantCLEF submission."""
    assert list(submission.columns) == ["quadrat_id", "species_ids"]
    assert len(submission) == expected_n_rows
    assert submission["quadrat_id"].isna().sum() == 0
    assert submission["species_ids"].isna().sum() == 0
    assert submission["species_ids"].str.startswith("[").all()
    assert submission["species_ids"].str.endswith("]").all()
    return True


def save_submission(submission: pd.DataFrame, output_path: str | Path) -> None:
    """Save a submission with Kaggle-compatible quoting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, sep=",", index=False, quoting=csv.QUOTE_ALL)
