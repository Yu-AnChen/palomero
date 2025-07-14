"""Core data structures for the Palomero application."""

import dataclasses
import numpy as np
from typing import Optional


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
    affine_only: bool
    from_mask_roi_id: Optional[int] = None
    to_mask_roi_id: Optional[int] = None
    row_num: Optional[int] = None  # For batch mode context


@dataclasses.dataclass
class AlignmentResult:
    """Result of a single alignment task."""

    image_id_from: int
    image_id_to: int
    success: bool
    message: str
    qc_plot_path: Optional[str] = None
    affine_matrix: Optional[np.ndarray] = None
    rois_mapped: Optional[int] = None
    row_num: Optional[int] = None
