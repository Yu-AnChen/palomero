"""Wraps Palom for coarse alignment."""

import logging
import itertools
import dask.array as da
import numpy as np
import palom
import palom.img_util
from palom.reader import DaPyramidChannelReader
from .. import omero_handler

log = logging.getLogger(__name__)


class PalomReaderFactory:
    """Factory to create Palom readers from Image Handlers."""

    @staticmethod
    def create_reader(
        handler: omero_handler.ImageHandler, channel: int, max_pixel_size: float
    ) -> DaPyramidChannelReader:
        """Creates a Palom DaPyramidChannelReader for a specific channel."""
        if not handler.is_valid():
            raise ValueError(f"ImageHandler for image {handler.image_id} is not valid.")

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
        pyramid = [
            da.zeros((1, *ss), chunks=1024, dtype=img.dtype, name=False)
            for ss in level_shapes
        ]
        pyramid[level] = img[np.newaxis]

        tip_pixel_size = pyramid_config["level_pixel_sizes"][level]
        while tip_pixel_size < max_pixel_size / 2:
            tip_pixel_size *= 2
            img = pyramid[-1][0]
            pyramid.append(palom.img_util.cv2_downscale_local_mean(img, 2)[np.newaxis])

        def level_downsamples(self):
            heights = [ss.shape[1] for ss in self.pyramid]
            heights.insert(0, heights[0])
            downsamples = [(h1 / h2) for h1, h2 in itertools.pairwise(heights)]
            return dict(enumerate(itertools.accumulate(downsamples, func=np.multiply)))

        palom.reader.DaPyramidChannelReader.level_downsamples = level_downsamples

        log.info(
            f"Creating Palom reader with {len(pyramid)} levels for image {handler.image_id}."
        )
        reader = DaPyramidChannelReader(pyramid, channel_axis=0)
        reader.level_downsamples = reader.level_downsamples()

        base_pixel_size = handler.base_pixel_size
        if base_pixel_size is None:
            log.warning(
                f"Using default pixel size 1.0 for Palom reader for image {handler.image_id}"
            )
            base_pixel_size = 1.0
        reader.pixel_size = base_pixel_size
        reader.path = handler.image_name
        reader.omero_metadata = {
            "image_id": handler.image_id,
            "channel": channel,
        }
        return reader


def run_coarse_alignment(
    reader1: DaPyramidChannelReader,
    reader2: DaPyramidChannelReader,
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
        _mx, _ = search_then_register(
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


def search_then_register(
    img_left,
    img_right,
    max_size=2000,
    n_keypoints=5000,
    auto_mask=True,
    plot_match_result=True,
    search_kwargs=None,
):
    from palom import img_util, register, register_util
    from palom.reader import logger
    from palom.register_dev import match_img_with_config, search_best_match_config

    search_kwargs = search_kwargs or {}
    img1 = img_left.astype("float32")
    img2 = img_right.astype("float32")

    shape_max = max(*img_left.shape, *img_right.shape)
    downsize_factor = int(np.ceil(shape_max / max_size))

    img1 = img_util.cv2_downscale_local_mean(img1, downsize_factor)
    img2 = img_util.cv2_downscale_local_mean(img2, downsize_factor)

    _, config = search_best_match_config(img1, img2, **search_kwargs)
    _img1, _img2 = match_img_with_config(
        img1,
        img2,
        img_util.entropy_mask(img1) if auto_mask else np.ones_like(img1, "bool"),
        img_util.entropy_mask(img2) if auto_mask else np.ones_like(img2, "bool"),
        *config,
    )
    mx, match = register.ensambled_match(
        _img1,
        _img2,
        n_keypoints=n_keypoints,
        plot_match_result=plot_match_result,
        return_match_mask=True,
        auto_invert_intensity=False,
        auto_mask=auto_mask,
    )
    mx_flip = np.eye(3)
    if config[2] == np.flipud:
        mx_flip = register_util.get_flip_mx(img2.shape, 0)
    if mx is None:
        logger.warning(
            "Feature matching failed. Returning identity matrix as placeholder"
        )
        mx = np.eye(3)[:2]
        match = np.zeros(1, "bool")
    mx = np.vstack([mx, [0, 0, 1]]) @ mx_flip

    def mx_scale(scale):
        mx = np.eye(3) * scale
        mx[2, 2] = 1
        return mx

    mx_full_res = mx_scale(downsize_factor) @ mx @ mx_scale(1 / downsize_factor)

    logger.debug(
        f"{match.sum():6} matches; {n_keypoints:6} keypoints; mask: {auto_mask}"
    )

    return mx_full_res[:2], config