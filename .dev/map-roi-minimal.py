# ezomero branch
# https://github.com/Yu-AnChen/ezomero/commit/9a2ab53d673e718cdf5a9dbae6ca37e9a335c815
# python -m pip install 'ezomero @ git+https://github.com/Yu-AnChen/ezomero@9a2ab53d673e718cdf5a9dbae6ca37e9a335c815'

import dataclasses
import logging
from functools import cached_property

import ezomero
import numpy as np
import palom
import tqdm
from omero.gateway import BlitzGateway
from typing import Optional


class ImageHandler:
    def __init__(self, conn: BlitzGateway, image_id: int):
        self.conn = conn
        self.image_id = image_id
        self.image_object = self.fetch_image_object()
        self.image_group_id = self.image_object.getDetails().getGroup().getId()
        set_session_group(self.conn, self.image_group_id)

    def fetch_image_object(self):
        self.conn.SERVICE_OPTS.setOmeroGroup("-1")
        # Load the image object once
        return self.conn.getObject("Image", self.image_id)

    @cached_property
    def num_channels(self):
        return len(self.image_object.getChannels())

    @property
    def num_pyramid_levels(self):
        return len(self.pyramid_config["level_shapes"])

    @property
    def pixel_size(self):
        return self.pyramid_config["level_pixel_sizes"][0]

    @cached_property
    def pyramid_config(self):
        set_session_group(self.conn, self.image_group_id)
        image = self.image_object

        pix = image._conn.c.sf.createRawPixelsStore()
        pid = image.getPixelsId()
        pix.setPixelsId(pid, False)
        level_shapes = [(rr.sizeY, rr.sizeX) for rr in pix.getResolutionDescriptions()]
        pix.close()

        _physical_size_x = image.getPrimaryPixels().getPhysicalSizeX()
        pixel_size = _physical_size_x.getValue()
        pixel_size_unit = _physical_size_x.getUnit()
        if pixel_size_unit.name != "MICROMETER":
            logging.warning(f"target image's pixel size unit is {pixel_size_unit}")

        level_downsamples = {
            kk: 1 / vv for kk, vv in image.getZoomLevelScaling().items()
        }
        level_pixel_sizes = [
            pixel_size * vv for _, vv in sorted(level_downsamples.items())
        ]
        return {
            "level_downsamples": [vv for _, vv in sorted(level_downsamples.items())],
            "level_pixel_sizes": level_pixel_sizes,
            "level_shapes": level_shapes,
        }

    def fetch_image_level_channel(self, level: int, channel: int):
        set_session_group(self.conn, self.image_group_id)

        _, img = ezomero.get_image(
            self.conn,
            self.image_id,
            pyramid_level=range(self.num_pyramid_levels)[level],
            start_coords=(0, 0, 0, channel, 0),
            axis_lengths=(*self.pyramid_config["level_shapes"][level][::-1], 1, 1, 1),
        )
        return img.squeeze()

    def select_pyramid_level_by_pixel_size(self, max_pixel_size: float):
        level_downsamples = self.pyramid_config["level_downsamples"]
        level_pixel_sizes = self.pyramid_config["level_pixel_sizes"]
        return np.max(
            np.arange(len(level_downsamples))[
                np.less_equal(level_pixel_sizes, max_pixel_size)
            ]
        )

    def channel_to_palom_reader(
        self, channel: int, max_pixel_size: float
    ) -> palom.reader.DaPyramidChannelReader:
        import itertools
        import dask.array as da
        import palom

        level = self.select_pyramid_level_by_pixel_size(max_pixel_size)
        img = self.fetch_image_level_channel(level, channel)

        pyramid = [
            da.zeros((1, *ss), chunks=1024, dtype=img.dtype, name=False)
            for ss in self.pyramid_config["level_shapes"][: level + 1]
        ]
        pyramid[level] = img[np.newaxis]

        tip_pixel_size = self.pyramid_config["level_pixel_sizes"][level]
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
        reader = palom.reader.DaPyramidChannelReader(pyramid, channel_axis=0)
        reader.level_downsamples = reader.level_downsamples()

        reader.pixel_size = self.pixel_size
        return reader


def set_group_by_image(conn: BlitzGateway, image_id: int) -> bool:
    conn.SERVICE_OPTS.setOmeroGroup("-1")
    image = conn.getObject("Image", image_id)
    if image is None:
        logging.warning(
            f"Cannot load image {image_id} - check if you have permissions to do so"
        )
        return False
    group_id = image.getDetails().getGroup().getId()
    return set_session_group(conn, group_id)


def set_session_group(conn: BlitzGateway, group_id: int) -> bool:
    current_id = conn.getGroupFromContext().getId()
    if group_id == current_id:
        return True
    if ezomero.set_group(conn, group_id):
        conn.setGroupForSession(group_id)
        return True
    return False


def _tform_mx(transform: ezomero.rois.AffineTransform) -> np.ndarray:
    return np.array(
        [
            [transform.a00, transform.a01, transform.a02],
            [transform.a10, transform.a11, transform.a12],
            [0, 0, 1],
        ]
    )


def map_roi(
    conn: BlitzGateway, from_image_id: int, to_image_id: int, affine_mx: np.ndarray
) -> list[ezomero.rois.ezShape]:
    if not conn.keepAlive():
        logging.error(f"Connection to {conn.host} lost")
        return
    set_group_by_image(conn, from_image_id)
    rois = ezomero.get_roi_ids(conn, from_image_id)
    shapes = [
        ezomero.get_shape(conn, ss)
        for rr in tqdm.tqdm(rois, "Downloading ROI")
        for ss in ezomero.get_shape_ids(conn, rr)
    ]
    transforms_ori = [ss.transform for ss in shapes]
    mxs = [_tform_mx(tt) if tt else np.eye(3) for tt in transforms_ori]
    transforms = [
        ezomero.rois.AffineTransform(*(affine_mx @ mm)[:2].T.ravel()) for mm in mxs
    ]
    shapes_transformed = [
        dataclasses.replace(ss, transform=tt) for ss, tt in zip(shapes, transforms)
    ]
    set_group_by_image(conn, to_image_id)
    for st in tqdm.tqdm(shapes_transformed, "Uploading ROI"):
        ezomero.post_roi(conn, to_image_id, [st])
    return shapes_transformed


def get_local_session(secure=True):
    import argparse

    from omero.plugins import sessions

    KEY = "79fc8d67-2dc9-4170-922d-62a0b58e5b3a"
    HOST = "omero-app.hms.harvard.edu"
    # URL = "https://omero.hms.harvard.edu"

    _parser = argparse.ArgumentParser(argument_default=True)
    _parser.add_argument("--purge")
    session_ctrl = sessions.SessionsControl()
    session_store = sessions.SessionsStore()

    # clear out inactive sessions
    session_ctrl.list(_parser.parse_args([]))

    HOST, USER, KEY, PORT = session_store.get_current()
    if KEY is None:
        msg = "No active session detected, please login using `omero login`"
        raise RuntimeError(msg)

    conn = ezomero.connect(KEY, KEY, host=HOST, port=PORT, secure=secure, group="")
    return conn


def test(do_post=False):
    import pathlib

    import matplotlib.pyplot as plt
    from palom.cli.align_he import save_all_figs, set_matplotlib_font, set_subplot_size

    qc_out_dir = "map-roi-qc"
    pathlib.Path(qc_out_dir).mkdir(exist_ok=True, parents=True)

    conn = get_local_session()

    ID_IF = 1614258
    ID_HE = 1623951

    r1 = ImageHandler(conn, ID_IF).channel_to_palom_reader(0, 50)
    r2 = ImageHandler(conn, ID_HE).channel_to_palom_reader(0, 50)

    c21l = palom.align.get_aligner(r1, r2)
    _mx = palom.register_dev.search_then_register(
        np.asarray(c21l.ref_thumbnail),
        np.asarray(c21l.moving_thumbnail),
        n_keypoints=10_000,
        auto_mask=True,
        max_size=2000,
    )
    c21l.coarse_affine_matrix = np.vstack([_mx, [0, 0, 1]])

    set_matplotlib_font(6)
    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle(f"{ID_HE} (coarse alignment)", fontsize=8)
    ax.set_title(f"{ID_IF} - {ID_HE}", fontsize=6)
    im_h, im_w = ax.images[0].get_array().shape
    set_subplot_size(im_w / 144, im_h / 144, ax=ax)
    ax.set_anchor("N")
    fig.subplots_adjust(top=1 - 0.5 / fig.get_size_inches()[1])
    save_all_figs(out_dir=qc_out_dir, format="jpg", dpi=144)

    print(c21l.affine_matrix)
    if do_post:
        _ = map_roi(conn, ID_HE, ID_IF, c21l.affine_matrix)


def align_omero_image_pair(
    conn: BlitzGateway,
    image_id_1: str,
    image_id_2: str,
    channel_1: int = 0,
    channel_2: int = 2,
    max_pixel_size: float = 50,
    n_keypoints: int = 10_000,
    auto_mask: bool = True,
    thumbnail_max_size: int = 2000,
    qc_out_dir: str = "map-roi-qc",
):
    import pathlib

    import matplotlib.pyplot as plt
    from palom.cli.align_he import save_all_figs, set_matplotlib_font, set_subplot_size

    pathlib.Path(qc_out_dir).mkdir(exist_ok=True, parents=True)

    r1 = ImageHandler(conn, image_id_1).channel_to_palom_reader(
        channel_1, max_pixel_size
    )
    r2 = ImageHandler(conn, image_id_2).channel_to_palom_reader(
        channel_2, max_pixel_size
    )

    c21l = palom.align.get_aligner(r1, r2)
    _mx = palom.register_dev.search_then_register(
        np.asarray(c21l.ref_thumbnail),
        np.asarray(c21l.moving_thumbnail),
        n_keypoints=n_keypoints,
        auto_mask=auto_mask,
        max_size=thumbnail_max_size,
    )
    c21l.coarse_affine_matrix = np.vstack([_mx, [0, 0, 1]])

    set_matplotlib_font(6)
    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle(f"{image_id_2} (coarse alignment)", fontsize=8)
    ax.set_title(f"{image_id_1} - {image_id_2}", fontsize=6)
    im_h, im_w = ax.images[0].get_array().shape
    set_subplot_size(im_w / 144, im_h / 144, ax=ax)
    ax.set_anchor("N")
    fig.subplots_adjust(top=1 - 0.5 / fig.get_size_inches()[1])
    save_all_figs(out_dir=qc_out_dir, format="jpg", dpi=144)

    return c21l.affine_matrix


def main():
    import argparse
    import pathlib
    import sys

    from loguru import logger

    parser = argparse.ArgumentParser(
        description="Align two OMERO images, generate QC plots, and optionally map ROIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-id-1",
        type=int,
        metavar="ID",
        help="OMERO Image ID for the first image (e.g., fixed).",
        required=True,
    )
    parser.add_argument(
        "--image-id-2",
        type=int,
        metavar="ID",
        help="OMERO Image ID for the second image (e.g., moving).",
        required=True,
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
        metavar="MASK",
        help="Automatically mask out background before image alignment. Default: %(default)s",
    )
    parser.add_argument(
        "--thumbnail-max-size",
        type=int,
        default=2000,
        metavar="SIZE",
        help="Max thumbnail size when determining image orientations. Default: %(default)s",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform alignment and QC plot generation, but DO NOT map/post ROIs.",
    )
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
    # --- Output Options ---
    parser.add_argument(
        "--qc-out-dir",
        type=str,
        default="map-roi-qc",
        metavar="DIR",
        help="Output directory for Quality Control (QC) alignment plots. Default: %(default)s",
    )

    args = parser.parse_args()
    print(args)
    args.image_id_1

    logger.remove()
    logger.add(sys.stderr)
    logger.add(pathlib.Path(args.qc_out_dir) / "map-roi.log", rotation="5 MB")

    conn: Optional[BlitzGateway] = None

    try:
        conn = get_local_session(args.secure)
        mx = align_omero_image_pair(
            conn=conn,
            image_id_1=args.image_id_1,
            image_id_2=args.image_id_2,
            channel_1=args.channel_1,
            channel_2=args.channel_2,
            max_pixel_size=args.max_pixel_size,
            n_keypoints=args.n_keypoints,
            auto_mask=args.auto_mask,
            thumbnail_max_size=args.thumbnail_max_size,
            qc_out_dir=args.qc_out_dir,
        )
        if not args.dry_run:
            map_roi(conn, args.image_id_2, args.image_id_1, mx)

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)  # Exit with error

    finally:
        # --- Disconnect ---
        if conn and conn.isConnected() and (not args.keepalive):
            try:
                conn.close()
                logger.info("OMERO connection closed.")
            except Exception as close_e:
                logger.warning(f"Error closing OMERO session: {close_e}")

    sys.exit(0)


# # --- Script Entry Point ---
if __name__ == "__main__":
    main()