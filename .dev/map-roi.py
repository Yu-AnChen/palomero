import argparse
import csv
import dataclasses
import inspect
import itertools
import logging
import os
import pathlib
import sys
import time
from functools import cached_property, wraps
from typing import Any, Dict, List, Optional

import ezomero
import numpy as np
import omero.model
import palom
import tqdm
from ezomero.rois import ezShape

# OMERO imports
from omero.gateway import BlitzGateway, ImageWrapper
from omero.plugins import sessions  # For standalone connection via session store

# --- Configuration --- Setup basic logging. Can be configured further in
# main().
log = logging.getLogger(__name__)  # Use __name__ for logger
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# --- Data Structures ---
@dataclasses.dataclass
class AlignmentTask:
    """Parameters for a single alignment task."""

    image_id_1: int
    image_id_2: int
    channel_1: int
    channel_2: int
    max_pixel_size: float
    n_keypoints: int
    auto_mask: bool
    thumbnail_max_size: int
    qc_out_dir: str
    map_rois: bool
    dry_run: bool
    row_num: Optional[int] = None  # For batch mode context


# AlignmentResult dataclass remains the same
@dataclasses.dataclass
class AlignmentResult:
    """Result of a single alignment task."""

    image_id_1: int
    image_id_2: int
    success: bool
    message: str
    qc_plot_path: Optional[str] = None
    affine_matrix: Optional[np.ndarray] = None
    rois_mapped: Optional[int] = None
    row_num: Optional[int] = None


# --- Error Handling Decorator ---
def robust_task_execution(func):
    """
    Decorator to wrap task execution, log errors, and ensure an AlignmentResult
    is always returned.
    """

    @wraps(func)
    def wrapper(
        conn: BlitzGateway, task: AlignmentTask, *args, **kwargs
    ) -> AlignmentResult:
        pair_label = f"{task.image_id_1}_vs_{task.image_id_2}"
        row_info = f"(CSV Row {task.row_num})" if task.row_num else ""
        log.debug(f"Executing task: {pair_label} {row_info}")
        try:
            result = func(conn, task, *args, **kwargs)
            if not isinstance(result, AlignmentResult):
                log.error(
                    f"Task function {func.__name__} did not return AlignmentResult for {pair_label}"
                )
                return AlignmentResult(
                    image_id_1=task.image_id_1,
                    image_id_2=task.image_id_2,
                    success=False,
                    message="Internal error: Invalid result type from task function.",
                    row_num=task.row_num,
                )
            return result
        except Exception as e:
            log.error(
                f"Task failed unexpectedly: {pair_label} {row_info} - {e}",
                exc_info=True,
            )
            return AlignmentResult(
                image_id_1=task.image_id_1,
                image_id_2=task.image_id_2,
                success=False,
                message=f"Unhandled exception: {e}",
                row_num=task.row_num,
            )

    return wrapper


# --- Helper Functions (Session, ROI Utils) ---
def set_session_group(conn: BlitzGateway, group_id: int) -> bool:
    """Sets the Omero session group context if it's not already set."""
    if not conn or not conn.keepAlive():
        log.error(f"Connection lost before setting group to {group_id}")
        return False
    try:
        current_id = conn.getGroupFromContext().getId()
        if group_id == current_id:
            log.debug(f"Session group already set to {group_id}.")
            return True
        conn.setGroupForSession(group_id)
        log.debug(f"Set session group to {group_id} via setGroupForSession.")
        return True
    except Exception as e:
        log.error(f"Error setting session group to {group_id}: {e}")
        return False


def set_group_by_image(conn: BlitzGateway, image_id: int) -> bool:
    """Sets the Omero session group context based on the image's group."""
    if not conn or not conn.keepAlive():
        log.error(f"Connection lost before setting group for image {image_id}")
        return False
    conn.SERVICE_OPTS.setOmeroGroup("-1")  # Ensure we can find the image across groups
    image = conn.getObject("Image", image_id)
    if image is None:
        log.warning(
            f"Cannot load image {image_id} to set group - check ID and permissions."
        )
        return False
    try:
        group_id = image.getDetails().getGroup().getId()
    except Exception as e:
        log.error(f"Could not get group details for image {image_id}: {e}")
        return False
    return set_session_group(conn, group_id)


def _tform_mx(transform: omero.model.AffineTransformI) -> np.ndarray:
    """Converts an omero AffineTransform object to a 3x3 numpy matrix."""
    if isinstance(transform, omero.model.AffineTransformI):
        return np.array(
            [
                [transform.a00.val, transform.a01.val, transform.a02.val],
                [transform.a10.val, transform.a11.val, transform.a12.val],
                [0, 0, 1],
            ]
        )
    elif hasattr(transform, "a00"):  # Duck typing for ezomero dataclass
        return np.array(
            [
                [transform.a00, transform.a01, transform.a02],
                [transform.a10, transform.a11, transform.a12],
                [0, 0, 1],
            ]
        )
    else:
        raise TypeError(f"Unsupported transform type for _tform_mx: {type(transform)}")


# --- Standalone Connection Function ---
def get_omero_connection(secure=True) -> Optional[BlitzGateway]:
    """
    Connects to Omero using the current active session from the Omero CLI
    sessions store.
    """
    log.info("Attempting to connect to OMERO using local session...")
    # Use a separate parser for session management to avoid conflicts with the
    # main script's parser if needed (though not strictly necessary here).
    session_parser = argparse.ArgumentParser(argument_default=True)
    session_parser.add_argument("--purge")  # Argument used by SessionsControl.list
    session_ctrl = sessions.SessionsControl()
    session_store = sessions.SessionsStore()

    log.debug("Checking Omero sessions...")
    try:
        # Suppress stdout from session_ctrl.list if possible, or just log around
        # it
        session_ctrl.list(session_parser.parse_args([]))  # Pass empty list
    except Exception as e:
        log.warning(f"Error during Omero session check: {e}")

    host, user, key, port = session_store.get_current()
    if key is None:
        log.error("No active Omero session detected. Please login using `omero login`.")
        return None

    log.info(f"Found active session: User={user}, Host={host}, Port={port}")

    try:
        conn = ezomero.connect(key, key, host=host, port=port, secure=secure, group="")
        if conn and conn.isConnected():
            log.info(f"Successfully connected to OMERO: {host}")
            return conn
        else:
            log.error(f"ezomero.connect failed for session: User={user}, Host={host}")
            if conn:
                conn.close()  # Ensure cleanup
            return None
    except Exception as e:
        log.error(
            f"Failed to connect to OMERO using session details: {e}", exc_info=True
        )
        return None


class ImageHandler:
    """
    Represents an OMERO Image and provides access to its metadata and data
    planes. Focuses on fetching and caching image properties.
    """

    # Using internal _conn to emphasize it's primarily for setup
    _conn: BlitzGateway
    _image_object: Optional[ImageWrapper]
    _group_id: Optional[int]

    def __init__(self, conn: BlitzGateway, image_id: int):
        self._conn = conn
        self.image_id = image_id
        self._image_object = self._fetch_image_object()
        self._group_id = None
        if self._image_object:
            try:
                details = self._image_object.getDetails()
                if details and details.group:
                    self._group_id = details.group.id.val
                    log.debug(
                        f"Initialized ImageHandler for Image ID: {self.image_id} in Group ID: {self._group_id}"
                    )
                else:
                    log.error(f"Failed to get group details for image {self.image_id}")
                    self._image_object = None  # Mark as invalid
            except Exception as e:
                log.error(f"Failed to get group ID for image {self.image_id}: {e}")
                self._image_object = None  # Mark as invalid

    def is_valid(self) -> bool:
        """Check if the image object was loaded successfully."""
        return self._image_object is not None

    def _fetch_image_object(self) -> Optional[ImageWrapper]:
        """Fetches the Omero image object."""
        if not self._conn or not self._conn.keepAlive():
            log.error(f"Omero connection lost before fetching image {self.image_id}")
            return None
        self._conn.SERVICE_OPTS.setOmeroGroup("-1")  # Search across groups
        try:
            image = self._conn.getObject("Image", self.image_id)
            if image is None:
                log.warning(
                    f"Cannot load image {self.image_id} - check ID and permissions."
                )
            return image
        except Exception as e:
            log.error(f"Error fetching image {self.image_id}: {e}")
            return None

    @property
    def group_id(self) -> Optional[int]:
        """The ID of the group this image belongs to."""
        return self._group_id

    @cached_property
    def num_channels(self) -> int:
        """Returns the number of channels (SizeC) in the image."""
        if not self.is_valid():
            return 0
        try:
            return self._image_object.getSizeC()
        except Exception as e:
            log.warning(f"Could not get SizeC for image {self.image_id}: {e}")
            return 0

    @cached_property
    def pyramid_config(self) -> Dict[str, Any]:
        """
        Calculates and caches pyramid configuration details. Returns: Dict with
        'level_downsamples', 'level_pixel_sizes', 'level_shapes'.
        """
        if not self.is_valid() or self.group_id is None:
            return {
                "level_downsamples": [],
                "level_pixel_sizes": [],
                "level_shapes": [],
            }

        set_session_group(self._conn, self.group_id)

        level_shapes = []
        pix = None
        try:
            pix = self._conn.c.sf.createRawPixelsStore()
            pixels_id = self._image_object.getPixelsId()
            pix.setPixelsId(pixels_id, False)
            level_shapes = [
                (rr.sizeY, rr.sizeX) for rr in pix.getResolutionDescriptions()
            ]
        except Exception as e:
            log.error(
                f"Failed to get resolution descriptions for image {self.image_id}: {e}"
            )
            if not level_shapes and self.is_valid():
                try:
                    shape_y = self._image_object.getSizeY()
                    shape_x = self._image_object.getSizeX()
                    level_shapes = [(shape_y, shape_x)]
                    log.warning(
                        f"Using primary SizeY/SizeX for image {self.image_id} as fallback shape."
                    )
                except Exception as fallback_e:
                    log.error(
                        f"Fallback shape fetch failed for image {self.image_id}: {fallback_e}"
                    )
        finally:
            if pix:
                pix.close()

        pixel_size = 1.0  # Default
        try:
            primary_pixels = self._image_object.getPrimaryPixels()
            if primary_pixels:
                _physical_size_x = primary_pixels.getPhysicalSizeX()
                if _physical_size_x:
                    pixel_size = _physical_size_x.getValue()
                    unit = _physical_size_x.getUnit().name
                    if unit != "MICROMETER":
                        log.warning(
                            f"Image {self.image_id} pixel size unit is {unit}, assuming micrometers."
                        )
                else:
                    log.warning(
                        f"Physical pixel size X not available for image {self.image_id}. Assuming 1.0 µm."
                    )
            else:
                log.warning(
                    f"Primary pixels object not found for image {self.image_id}. Assuming 1.0 µm pixel size."
                )
        except Exception as e:
            log.warning(
                f"Could not get physical pixel size X for image {self.image_id}: {e}. Assuming 1.0 µm."
            )

        level_downsamples_map = {}
        try:
            zoom_scaling = self._image_object.getZoomLevelScaling()
            if zoom_scaling:
                level_downsamples_map = {kk: 1 / vv for kk, vv in zoom_scaling.items()}
            else:
                log.warning(
                    f"No zoom level scaling found for image {self.image_id}. Assuming single level."
                )
                if level_shapes:
                    level_downsamples_map = {0: 1}
        except Exception as e:
            log.warning(
                f"Could not get zoom level scaling for image {self.image_id}: {e}. Assuming single level."
            )
            if level_shapes and not level_downsamples_map:
                level_downsamples_map = {i: 1 for i in range(len(level_shapes))}

        level_pixel_sizes = [
            pixel_size * vv for _, vv in sorted(level_downsamples_map.items())
        ]

        num_levels = len(level_shapes)
        config = {
            "level_downsamples": [
                vv for _, vv in sorted(level_downsamples_map.items())
            ][:num_levels],
            "level_pixel_sizes": level_pixel_sizes[:num_levels],
            "level_shapes": level_shapes,
        }
        log.debug(f"Pyramid config for image {self.image_id}: {config}")
        return config

    @property
    def num_pyramid_levels(self) -> int:
        """Returns the number of pyramid levels available based on shapes."""
        return len(self.pyramid_config.get("level_shapes", []))

    @property
    def base_pixel_size(self) -> Optional[float]:
        """Returns the pixel size of the base level (level 0), if available."""
        sizes = self.pyramid_config.get("level_pixel_sizes", [])
        return sizes[0] if sizes else None

    def select_pyramid_level_by_pixel_size(self, max_pixel_size: float) -> int:
        """
        Selects the index of the highest resolution pyramid level whose pixel
        size is less than or equal to the specified maximum.
        """
        if not self.is_valid():
            return 0
        level_pixel_sizes = np.array(self.pyramid_config.get("level_pixel_sizes", []))
        if level_pixel_sizes.size == 0:
            log.warning(
                f"No level pixel sizes available for image {self.image_id}. Using level 0."
            )
            return 0

        valid_levels = np.where(level_pixel_sizes <= max_pixel_size)[0]
        selected_level: int
        if len(valid_levels) == 0:
            selected_level = 0
            log.warning(
                f"No pyramid level found with pixel size <= {max_pixel_size} µm "
                f"for image {self.image_id}. Using level 0 "
                f"(pixel size: {level_pixel_sizes[0]:.2f} µm)."
            )
        else:
            selected_level = int(np.max(valid_levels))
        log.info(
            f"Selected level {selected_level} for image {self.image_id} "
            f"(pixel size: {level_pixel_sizes[selected_level]:.2f} µm <= {max_pixel_size} µm)"
        )
        return selected_level

    def fetch_image_level_channel(self, level: int, channel: int) -> np.ndarray:
        """Fetches a specific 2D image plane for a given pyramid level and
        channel."""
        if not self.is_valid() or self.group_id is None:
            raise ValueError(
                f"Image handler for {self.image_id} is not valid or missing group ID."
            )

        set_session_group(self._conn, self.group_id)

        num_levels = self.num_pyramid_levels
        num_ch = self.num_channels
        if not (0 <= level < num_levels):
            raise IndexError(
                f"Level {level} out of bounds (0-{num_levels - 1}) for image {self.image_id}"
            )
        if not (0 <= channel < num_ch):
            raise IndexError(
                f"Channel {channel} out of bounds (0-{num_ch - 1}) for image {self.image_id}"
            )

        level_index_omero = level
        shape_yx = self.pyramid_config["level_shapes"][level]

        log.debug(
            f"Fetching Image: {self.image_id}, Level: {level} (Omero Index: {level_index_omero}), Channel: {channel}, Shape: {shape_yx}"
        )

        try:
            _, img = ezomero.get_image(
                conn=self._conn,
                image_id=self.image_id,
                pyramid_level=level_index_omero,
                start_coords=(0, 0, 0, channel, 0),
                axis_lengths=(*shape_yx[::-1], 1, 1, 1),
                across_groups=False,
            )
            return np.squeeze(img)
        except Exception as e:
            log.error(
                f"ezomero.get_image failed for image {self.image_id}, level {level}, channel {channel}: {e}"
            )
            raise RuntimeError(f"Failed to fetch image plane: {e}") from e


class PalomReaderFactory:
    """Factory to create Palom readers from Image Handlers."""

    @staticmethod
    def create_reader(
        handler: ImageHandler, channel: int, max_pixel_size: float
    ) -> palom.reader.DaPyramidChannelReader:
        """Creates a Palom DaPyramidChannelReader for a specific channel."""
        if not handler.is_valid():
            raise ValueError(f"ImageHandler for image {handler.image_id} is not valid.")

        import dask.array as da
        import palom.img_util

        level = handler.select_pyramid_level_by_pixel_size(max_pixel_size)
        try:
            img = handler.fetch_image_level_channel(level, channel)
        except Exception as e:
            log.error(
                f"Failed fetching data for Palom reader creation (Image: {handler.image_id}, Level: {level}, Channel: {channel}): {e}"
            )
            raise ValueError(
                f"Could not fetch image data for reader creation: {e}"
            ) from e

        pyramid_config = handler.pyramid_config
        if not pyramid_config or not pyramid_config["level_shapes"]:
            raise ValueError(
                f"Missing pyramid configuration for image {handler.image_id}"
            )

        level_shapes = pyramid_config["level_shapes"][: level + 1]
        pyramid: List[da.Array] = []
        try:
            pyramid = [
                da.zeros((1, *ss), chunks=1024, dtype=img.dtype, name=False)
                for ss in level_shapes
            ]
        except TypeError as te:
            log.error(
                f"Error creating dask pyramid structure for image {handler.image_id}: {te}. Shapes: {level_shapes}"
            )
            raise ValueError(
                f"Invalid level shapes for image {handler.image_id}"
            ) from te

        pyramid[level] = img[np.newaxis]

        tip_pixel_size = pyramid_config["level_pixel_sizes"][level]
        while tip_pixel_size < max_pixel_size / 2:
            tip_pixel_size *= 2
            img = pyramid[-1][0]
            pyramid.append(palom.img_util.cv2_downscale_local_mean(img, 2)[np.newaxis])

        # FIXME monkey patching for palom's level downsample calculation
        def level_downsamples(self):
            heights = [ss.shape[1] for ss in self.pyramid]
            heights.insert(0, heights[0])
            downsamples = [(h1 / h2) for h1, h2 in itertools.pairwise(heights)]
            return dict(enumerate(itertools.accumulate(downsamples, func=np.multiply)))

        palom.reader.DaPyramidChannelReader.level_downsamples = level_downsamples

        log.info(
            f"Creating Palom reader with {len(pyramid)} levels for image {handler.image_id}."
        )
        reader = palom.reader.DaPyramidChannelReader(pyramid, channel_axis=0)
        reader.level_downsamples = reader.level_downsamples()

        base_pixel_size = handler.base_pixel_size
        if base_pixel_size is None:
            log.warning(
                f"Using default pixel size 1.0 for Palom reader for image {handler.image_id}"
            )
            base_pixel_size = 1.0
        reader.pixel_size = base_pixel_size
        return reader


def _fetch_and_transform_rois(
    conn: BlitzGateway, from_image_id: int, affine_mx: np.ndarray
) -> List[ezShape]:
    """Fetches ROIs/Shapes from source image and applies transformation."""
    log.info(f"Fetching and transforming ROIs from Image {from_image_id}")
    if not set_group_by_image(conn, from_image_id):
        raise RuntimeError(
            f"Could not set context to source image {from_image_id} group."
        )

    shapes_transformed: List[ezShape] = []
    try:
        roi_ids = ezomero.get_roi_ids(conn, from_image_id)
        if not roi_ids:
            log.info(f"No ROIs found on source image {from_image_id}.")
            return []
        log.info(f"Found {len(roi_ids)} ROIs on source image {from_image_id}.")

        shapes: List[ezShape] = []
        for rr in tqdm.tqdm(
            roi_ids, desc=f"Downloading Shapes from {from_image_id}", unit="ROI"
        ):
            shape_ids = ezomero.get_shape_ids(conn, rr)
            if not shape_ids:
                continue
            for ss_id in shape_ids:
                try:
                    shape = ezomero.get_shape(conn, ss_id)
                    if shape:
                        shapes.append(shape)
                except Exception as e:
                    log.warning(
                        f"Could not get shape ID {ss_id} from ROI {rr} on image {from_image_id}: {e}"
                    )
        log.info(f"Downloaded {len(shapes)} shapes.")
        if not shapes:
            return []

        log.info("Applying affine transformation to shapes...")
        for ss in shapes:
            original_transform = ss.transform
            original_mx = (
                _tform_mx(original_transform) if original_transform else np.eye(3)
            )
            combined_mx = affine_mx @ original_mx
            combined_transform = ezomero.rois.AffineTransform(
                *combined_mx[:2].T.ravel()
            )
            tformed_ss = dataclasses.replace(ss, transform=combined_transform)
            shapes_transformed.append(tformed_ss)

    except Exception as e:
        log.error(
            f"Error fetching or transforming ROIs from image {from_image_id}: {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Failed during ROI fetch/transform: {e}") from e

    log.info(f"Successfully transformed {len(shapes_transformed)} shapes.")
    return shapes_transformed


def _post_rois(conn: BlitzGateway, to_image_id: int, shapes: List[ezShape]) -> int:
    """Posts transformed shapes (expected ezShape dataclasses) to the target
    image."""
    if not shapes:
        log.info("No shapes provided for posting.")
        return 0

    log.info(f"Uploading {len(shapes)} transformed shapes to image {to_image_id}...")
    if not set_group_by_image(conn, to_image_id):
        raise RuntimeError(
            f"Could not set context to target image {to_image_id} group."
        )

    posted_shapes_count = 0
    for ss in tqdm.tqdm(
        shapes, desc=f"Uploading Shapes to {to_image_id}", unit="shape"
    ):
        try:
            new_roi = ezomero.post_roi(conn, to_image_id, [ss])
            if new_roi:
                posted_shapes_count += 1
            else:
                log.warning(
                    "Failed to post shape (ezomero.post_roi returned None or raised error)"
                )
        except Exception as e:
            shape_info = f"type {type(ss).__name__}" if ss else "N/A"
            log.warning(f"Failed to post shape ({shape_info}): {e}", exc_info=True)

    log.info(
        f"Successfully posted {posted_shapes_count} shapes to image {to_image_id}."
    )
    return posted_shapes_count


def _perform_palom_alignment(
    reader1: palom.reader.OmePyramidReader,
    reader2: palom.reader.OmePyramidReader,
    n_keypoints: int,
    auto_mask: bool,
    max_size: int,
) -> palom.align.Aligner:
    """Performs coarse affine registration using Palom."""
    log.info("Performing coarse affine registration...")
    if reader1.pixel_size is None or reader2.pixel_size is None:
        raise ValueError("Palom readers must have pixel_size attribute set.")

    aligner = palom.align.get_aligner(reader1, reader2)
    try:
        _mx = palom.register_dev.search_then_register(
            np.asarray(aligner.ref_thumbnail),
            np.asarray(aligner.moving_thumbnail),
            n_keypoints=n_keypoints,
            auto_mask=auto_mask,
            max_size=max_size,
        )
        aligner.coarse_affine_matrix = np.vstack([_mx, [0, 0, 1]])
        log.info(
            f"Coarse alignment successful. Affine matrix (Reader1 -> Reader2):\n{aligner.affine_matrix}"
        )
        return aligner
    except Exception as e:
        log.error(f"Palom coarse alignment failed: {e}", exc_info=True)
        raise RuntimeError(f"Palom alignment failed: {e}") from e


def _generate_qc_plot(task: AlignmentTask) -> Optional[str]:
    """Generates and saves the QC plot."""
    log.info("Generating QC plot...")
    fig = None
    plot_path = None
    try:
        import matplotlib.pyplot as plt
        from palom.cli.align_he import set_matplotlib_font, set_subplot_size

        set_matplotlib_font(6)

        fig, ax = plt.gcf(), plt.gca()

        fig.suptitle(
            f"Coarse Alignment QC: {task.image_id_1} (Ch {task.channel_1}) vs {task.image_id_2} (Ch {task.channel_2})",
            fontsize=10,
        )
        ax.set_title(
            f"Max Pixel Size: {task.max_pixel_size} µm, Keypoints: {task.n_keypoints}\n"
            f"Auto Mask Tissue: {task.auto_mask}, Max Thumbnail Size: {task.thumbnail_max_size}",
            fontsize=8,
        )
        im_h, im_w = ax.images[0].get_array().shape
        set_subplot_size(im_w / 144, im_h / 144, ax=ax)
        ax.set_anchor("N")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        pair_label = f"{task.image_id_1}_vs_{task.image_id_2}"
        qc_filename_prefix = f"qc_alignment_{pair_label}"
        qc_dir = pathlib.Path(task.qc_out_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / qc_filename_prefix
        plot_path = str(qc_path.with_suffix(".jpg"))

        fig.savefig(plot_path, dpi=144, bbox_inches="tight")
        log.info(f"QC plot saved to '{plot_path}'")
        return plot_path
    except Exception as e:
        log.error(f"Failed to generate or save QC plot: {e}", exc_info=True)
        return None
    finally:
        if fig and "plt" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


# Apply decorator here
@robust_task_execution
def run_single_alignment_task(
    conn: BlitzGateway, task: AlignmentTask
) -> AlignmentResult:
    """
    Orchestrates the alignment and mapping for a single pair of images.
    Decorated for robust error handling. Returns AlignmentResult.
    """
    pair_label = f"{task.image_id_1}_vs_{task.image_id_2}"
    log.info(f"--- Starting task: {pair_label} ---")
    qc_plot_path: Optional[str] = None
    affine_matrix: Optional[np.ndarray] = None
    rois_mapped: Optional[int] = None
    message: str = ""
    success = False  # Default to failure unless explicitly set

    # 1. Setup Image Handlers and Palom Readers
    log.debug(f"[{pair_label}] Initializing handlers and readers...")
    handler1 = ImageHandler(conn, task.image_id_1)
    handler2 = ImageHandler(conn, task.image_id_2)
    if not handler1.is_valid():
        raise ValueError(
            f"Image handler failed to initialize for Image ID: {task.image_id_1}"
        )
    if not handler2.is_valid():
        raise ValueError(
            f"Image handler failed to initialize for Image ID: {task.image_id_2}"
        )

    reader1 = PalomReaderFactory.create_reader(
        handler1, task.channel_1, task.max_pixel_size
    )
    reader2 = PalomReaderFactory.create_reader(
        handler2, task.channel_2, task.max_pixel_size
    )

    # 2. Perform Alignment
    log.debug(f"[{pair_label}] Performing alignment...")
    aligner = _perform_palom_alignment(
        reader1,
        reader2,
        n_keypoints=task.n_keypoints,
        auto_mask=task.auto_mask,
        max_size=task.thumbnail_max_size,
    )
    affine_matrix = aligner.affine_matrix

    # 3. Generate QC Plot
    log.debug(f"[{pair_label}] Generating QC plot...")
    qc_plot_path = _generate_qc_plot(task)

    # 4. Map ROIs (if requested and not dry run)
    roi_mapping_message = ""
    if task.map_rois and not task.dry_run:
        log.info(
            f"[{pair_label}] Mapping ROIs from {task.image_id_2} to {task.image_id_1}..."
        )
        try:
            transformed_shapes = _fetch_and_transform_rois(
                conn, task.image_id_2, affine_matrix
            )
            rois_mapped = _post_rois(conn, task.image_id_1, transformed_shapes)
            roi_mapping_message = f" Mapped {rois_mapped} ROIs."
        except Exception as roi_e:
            log.error(
                f"[{pair_label}] Error during ROI mapping step: {roi_e}", exc_info=True
            )
            raise RuntimeError(f"ROI mapping failed: {roi_e}") from roi_e

    # If alignment succeeded, mark as success
    success = True
    message = "Alignment successful." + roi_mapping_message
    if task.dry_run and task.map_rois:
        message += " (Dry run: ROI mapping skipped)."
    elif not task.map_rois:
        message += " (ROI mapping not requested)."

    return AlignmentResult(
        image_id_1=task.image_id_1,
        image_id_2=task.image_id_2,
        success=success,
        message=message,
        qc_plot_path=qc_plot_path,
        affine_matrix=affine_matrix,
        rois_mapped=rois_mapped,
        row_num=task.row_num,
    )


# --- Main Function (for standalone execution) ---


def prepare_batch_tasks(args: argparse.Namespace) -> List[AlignmentTask]:
    """Reads CSV using csv module and prepares list of AlignmentTask objects."""
    # This function is similar to _prepare_batch_tasks in the plugin, but
    # standalone
    log.info(f"Reading batch tasks from: {args.batch_csv}")
    tasks: List[AlignmentTask] = []
    required_headers = ["image_id_1", "image_id_2"]
    task_annot = dict(
        filter(
            lambda x: (x[0] not in required_headers)
            & (x[1] in [str, bool, int, float]),
            inspect.get_annotations(AlignmentTask).items(),
        )
    )
    try:
        with open(args.batch_csv, mode="r", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                raise ValueError("CSV file appears to be empty or has no header.")
            if not all(h in reader.fieldnames for h in required_headers):
                missing = set(required_headers) - set(reader.fieldnames)
                raise ValueError(f"CSV missing required headers: {', '.join(missing)}")

            for i, row in enumerate(reader):
                row_num = i + 2  # Account for header row and 0-based index
                try:
                    kwargs = {
                        "image_id_1": int(row["image_id_1"]),
                        "image_id_2": int(row["image_id_2"]),
                    }
                    for kk, vv in task_annot.items():
                        kwargs[kk] = vv(row.get(kk) or getattr(args, kk))
                    kwargs["row_num"] = row_num

                    tasks.append(AlignmentTask(**kwargs))
                except (ValueError, TypeError, KeyError) as ve:
                    log.warning(
                        f"Skipping CSV row {row_num} due to invalid value or missing key: {ve}. Row data: {row}"
                    )
                    continue

        log.info(f"Prepared {len(tasks)} tasks from CSV file.")
        return tasks
    except FileNotFoundError:
        log.error(f"Batch CSV file not found: {args.batch_csv}")
        raise  # Re-raise for main to handle
    except Exception as e:
        log.error(f"Failed to read or parse CSV file {args.batch_csv}: {e}")
        raise  # Re-raise


def report_summary(
    successful_results: List[AlignmentResult],
    failed_results: List[AlignmentResult],
    duration: float,
):
    """Prints the final summary to the console."""
    # This function is similar to _report_summary in the plugin, but uses
    # print/log
    total_tasks = len(successful_results) + len(failed_results)
    print("\n--- Processing Summary ---")
    print(f"Total tasks attempted: {total_tasks}")
    print(f"Successful tasks: {len(successful_results)}")
    print(f"Failed tasks: {len(failed_results)}")
    if failed_results:
        log.warning("Failures occurred:")
        print("\nFailures occurred:", file=sys.stderr)
        failed_results.sort(
            key=lambda r: r.row_num if r.row_num is not None else float("inf")
        )
        for result in failed_results:
            pair_label = f"{result.image_id_1}_vs_{result.image_id_2}"
            row_info = f"(CSV Row {result.row_num})" if result.row_num else ""
            log.warning(f"  - Pair {pair_label} {row_info}: {result.message}")
            print(
                f"  - Pair {pair_label} {row_info}: {result.message}", file=sys.stderr
            )
    print(f"\nTotal execution time: {duration:.2f} seconds")


def main():
    os.environ["COLUMNS"] = "80"

    """Main execution function for the standalone script."""
    parser = argparse.ArgumentParser(
        description="Align two OMERO images or a batch from CSV (sequentially), generate QC plots, and optionally map ROIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input Mode ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--batch-csv",
        metavar="FILE",
        help="Path to CSV file for batch processing (processed sequentially). "
        "Required columns: image_id_1, image_id_2, channel_1, channel_2.",
    )
    mode_group.add_argument(
        "--image-id-1",
        type=int,
        metavar="ID",
        help="Omero Image ID for the first image (e.g., reference/IF).",
    )

    parser.add_argument(
        "--image-id-2",
        type=int,
        metavar="ID",
        help="Omero Image ID for the second image (e.g., target/HE). Required if --image-id-1 is provided.",
    )

    # --- Alignment and Mapping Parameters ---
    parser.add_argument(
        "--channel-1",
        type=int,
        default=0,
        metavar="CH",
        help="Default channel index for the first image. Default: %(default)s",
    )
    parser.add_argument(
        "--channel-2",
        type=int,
        default=0,
        metavar="CH",
        help="Default channel index for the second image. Default: %(default)s",
    )
    parser.add_argument(
        "--max-pixel-size",
        type=float,
        default=50.0,
        metavar="MICRONS",
        help="Maximum pixel size for selecting pyramid level for alignment. Default: %(default)s µm",
    )
    parser.add_argument(
        "--n-keypoints",
        type=int,
        default=10_000,
        metavar="N",
        help="Number of keypoints for Palom SIFT feature detection. Default: %(default)s",
    )
    parser.add_argument(
        "--auto-mask",
        type=bool,
        default=True,
        metavar="DO_MASK",
        help="Automatically mask out background before image alignment. Default: %(default)s",
    )
    parser.add_argument(
        "--thumbnail-max-size",
        type=int,
        default=2000,
        metavar="MAX_SIZE",
        help="Max thumbnail size when determining image orientations. Default: %(default)s",
    )
    parser.add_argument(
        "--map-rois",
        action="store_true",
        help="Attempt to map ROIs from image_id_2 to image_id_1. Ignored if --dry-run is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform alignment and QC plot generation, but DO NOT map/post ROIs.",
    )

    # --- Output Options ---
    parser.add_argument(
        "--qc-out-dir",
        type=str,
        default="map-roi-qc",
        metavar="DIR",
        help="Output directory for Quality Control (QC) alignment plots. Default: %(default)s",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )

    # --- Connection Options ---
    parser.add_argument(
        "--secure",
        type=bool,
        default=True,
        help="Use secure session. Default: %(default)s",
    )
    parser.add_argument(
        "--keepalive",
        type=bool,
        default=True,
        help="Do not logout before exiting. Default: %(default)s",
    )

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.image_id_1 is not None and args.image_id_2 is None:
        parser.error("--image-id-2 is required when --image-id-1 is provided.")

    # --- Configure Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )  # force=True to override basicConfig if already called

    # --- Get OMERO Connection ---
    conn: Optional[BlitzGateway] = None
    successful_results: List[AlignmentResult] = []
    failed_results: List[AlignmentResult] = []
    start_time = time.time()

    try:
        conn = get_omero_connection(secure=args.secure)
        if not conn:
            # Error already logged by get_omero_connection
            print("Exiting due to OMERO connection failure.", file=sys.stderr)
            sys.exit(1)

        # --- Execute based on mode ---
        if args.batch_csv:
            tasks = prepare_batch_tasks(args)  # Can raise errors
            total_tasks = len(tasks)
            if total_tasks == 0:
                log.info("No valid tasks found in CSV. Exiting.")
            else:
                log.info(
                    f"Starting sequential batch processing for {total_tasks} task(s)."
                )
                for task in tqdm.tqdm(tasks, desc="Processing Batch Sequentially"):
                    if not conn.keepAlive():
                        log.error(
                            "OMERO connection lost during batch processing. Aborting."
                        )
                        failed_results.append(
                            AlignmentResult(
                                image_id_1=task.image_id_1,
                                image_id_2=task.image_id_2,
                                success=False,
                                message="OMERO connection lost.",
                                row_num=task.row_num,
                            )
                        )
                        break  # Stop processing
                    result = run_single_alignment_task(conn, task)
                    if result.success:
                        successful_results.append(result)
                    else:
                        failed_results.append(result)
        else:
            # Single mode
            if args.image_id_1 is None or args.image_id_2 is None:
                parser.error(
                    "Internal error: image_id_1 or image_id_2 missing for single mode."
                )  # Should be caught earlier
            log.info(
                f"Starting Single Pair Mode for: {args.image_id_1} vs {args.image_id_2}"
            )
            task = AlignmentTask(
                image_id_1=args.image_id_1,
                image_id_2=args.image_id_2,
                channel_1=args.channel_1,
                channel_2=args.channel_2,
                max_pixel_size=args.max_pixel_size,
                n_keypoints=args.n_keypoints,
                auto_mask=args.auto_mask,
                thumbnail_max_size=args.thumbnail_max_size,
                qc_out_dir=args.qc_out_dir,
                map_rois=args.map_rois,
                dry_run=args.dry_run,
                row_num=None,
            )
            result = run_single_alignment_task(conn, task)
            if result.success:
                successful_results.append(result)
            else:
                failed_results.append(result)

    except Exception as e:
        # Catch errors during setup (like CSV reading) or unexpected errors
        log.error(f"Script execution failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with error
    finally:
        # --- Disconnect ---
        if conn and conn.isConnected() and (not args.keepalive):
            try:
                conn.close()
                log.info("OMERO connection closed.")
            except Exception as close_e:
                log.warning(f"Error closing OMERO session: {close_e}")

    # --- Report Summary ---
    duration = time.time() - start_time
    report_summary(successful_results, failed_results, duration)

    # --- Exit Status ---
    if failed_results:
        sys.exit(1)
    else:
        sys.exit(0)


# --- Script Entry Point ---
if __name__ == "__main__":
    main()