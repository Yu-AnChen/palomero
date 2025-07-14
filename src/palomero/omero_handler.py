"""Handles all direct communication with the OMERO server."""

import argparse
import dataclasses
import logging
from functools import cached_property
from typing import Any, Dict, List, Optional

import ezomero
import numpy as np
import omero.model
import tqdm
from ezomero.rois import ezShape
from omero.gateway import BlitzGateway, ImageWrapper
from omero.plugins import sessions

log = logging.getLogger(__name__)


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


def get_omero_connection(secure=True) -> Optional[BlitzGateway]:
    """
    Connects to Omero using the current active session from the Omero CLI
    sessions store.
    """
    log.info("Attempting to connect to OMERO using local session...")
    session_parser = argparse.ArgumentParser(argument_default=True)
    session_parser.add_argument("--purge")
    session_ctrl = sessions.SessionsControl()
    session_store = sessions.SessionsStore()

    log.debug("Checking Omero sessions...")
    try:
        session_ctrl.list(session_parser.parse_args([]))
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
                conn.close()
            return None
    except Exception as e:
        log.error(
            f"Failed to connect to OMERO using session details: {e}", exc_info=True
        )
        return None


class ImageHandler:
    """
    Represents an OMERO Image and provides access to its metadata and data
    planes.
    """

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
                    self._image_object = None
            except Exception as e:
                log.error(f"Failed to get group ID for image {self.image_id}: {e}")
                self._image_object = None

    def is_valid(self) -> bool:
        return self._image_object is not None

    def _fetch_image_object(self) -> Optional[ImageWrapper]:
        if not self._conn or not self._conn.keepAlive():
            log.error(f"Omero connection lost before fetching image {self.image_id}")
            return None
        self._conn.SERVICE_OPTS.setOmeroGroup("-1")
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
        return self._group_id

    @cached_property
    def image_name(self) -> str:
        if not self.is_valid():
            return f"Image-{self.image_id}"
        try:
            return self._image_object.getName() or f"Image-{self.image_id}"
        except Exception as e:
            log.warning(f"Could not get name for image {self.image_id}: {e}")
            return f"Image-{self.image_id}"

    @cached_property
    def num_channels(self) -> int:
        if not self.is_valid():
            return 0
        try:
            return self._image_object.getSizeC()
        except Exception as e:
            log.warning(f"Could not get SizeC for image {self.image_id}: {e}")
            return 0

    @cached_property
    def pyramid_config(self) -> Dict[str, Any]:
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

        pixel_size = 1.0
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
        return len(self.pyramid_config.get("level_shapes", []))

    @property
    def base_pixel_size(self) -> Optional[float]:
        sizes = self.pyramid_config.get("level_pixel_sizes", [])
        return sizes[0] if sizes else None

    def select_pyramid_level_by_pixel_size(self, max_pixel_size: float) -> int:
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


def _get_shapes_from_rois(conn: BlitzGateway, roi_ids: List[int]) -> List[ezShape]:
    """Fetches all shapes from a list of OMERO ROI IDs."""
    shapes: List[ezShape] = []
    if not roi_ids:
        return shapes

    for r_id in tqdm.tqdm(roi_ids, desc="Downloading Shapes", unit="ROI"):
        try:
            shape_ids = ezomero.get_shape_ids(conn, r_id)
            if not shape_ids:
                continue
            for s_id in shape_ids:
                try:
                    shape = ezomero.get_shape(conn, s_id)
                    if shape:
                        shapes.append(shape)
                except Exception as e:
                    log.warning(f"Could not get shape ID {s_id} from ROI {r_id}: {e}")
        except Exception as e:
            log.warning(f"Could not get shape IDs from ROI {r_id}: {e}")
    return shapes


def get_shapes_from_roi(conn: BlitzGateway, roi_id: int) -> List[ezShape]:
    """Fetches all shapes from a specific OMERO ROI."""
    log.info(f"Fetching shapes from ROI {roi_id}")
    try:
        # We need to find which image this ROI belongs to set the group context
        roi = conn.getObject("Roi", roi_id)
        if not roi:
            raise ValueError(f"ROI with ID {roi_id} not found.")
        image_id = roi.getImage().getId()
        if not set_group_by_image(conn, image_id):
            raise RuntimeError(f"Could not set context to image {image_id} group.")

        shapes = _get_shapes_from_rois(conn, [roi_id])
        log.info(f"Downloaded {len(shapes)} shapes from ROI {roi_id}.")
        return shapes
    except Exception as e:
        log.error(f"Error fetching shapes from ROI {roi_id}: {e}", exc_info=True)
        raise RuntimeError(f"Failed during shape fetch for ROI {roi_id}: {e}") from e


def fetch_and_transform_rois(
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

        shapes = _get_shapes_from_rois(conn, roi_ids)
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


def post_rois(conn: BlitzGateway, to_image_id: int, shapes: List[ezShape]) -> int:
    """Posts transformed shapes to the target image."""
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
