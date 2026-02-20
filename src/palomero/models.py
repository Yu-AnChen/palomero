"""Core data structures for the Palomero application."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class AlignmentTask:
    """Parameters for a single alignment task."""

    image_id_from: int
    image_id_to: int
    channel_from: int
    channel_to: int
    max_pixel_size: float
    n_keypoints: int
    auto_mask: bool
    thumbnail_max_size: int
    qc_out_dir: str
    map_rois: bool
    dry_run: bool
    only_affine: bool
    sample_size_factor: float
    mask_roi_id_from: int | None = None
    mask_roi_id_to: int | None = None
    row_num: int | None = None  # For batch mode context


@dataclasses.dataclass
class AlignmentResult:
    """Result of a single alignment task."""

    image_id_from: int
    image_id_to: int
    success: bool
    message: str
    qc_plot_path: str | None = None
    affine_matrix: np.ndarray | None = None
    rois_mapped: int | None = None
    row_num: int | None = None
