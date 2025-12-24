"""Main alignment orchestrator class."""

import abc
import itertools
import logging
import pathlib
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import palom
import skimage.exposure
import skimage.morphology
import skimage.transform
import skimage.util

from .. import omero_handler, transform_roi_points
from ..models import AlignmentTask
from . import elastix_wrapper, palom_wrapper

log = logging.getLogger(__name__)


# --- Data Structures ---
@dataclass
class AffineResult:
    ref_img: np.ndarray
    affine_moving_img: np.ndarray
    affine_moving_mask: np.ndarray
    affine_matrix: np.ndarray


@dataclass
class ElastixResult:
    ref_img: np.ndarray
    affine_moving_img: np.ndarray
    elastix_moving_img: np.ndarray
    transform_params: dict
    error: Exception = None


# --- Alignment Strategies ---
class AlignmentStrategy(abc.ABC):
    def __init__(self, task: AlignmentTask):
        self.task = task

    @abc.abstractmethod
    def align(self, reader_ref, reader_moving, plot=False):
        pass


class AffineAligner(AlignmentStrategy):
    def align(self, reader_ref, reader_moving, plot=False):
        aligner = palom.align.get_aligner(reader_ref, reader_moving)
        _mx, intensity_config = palom_wrapper.search_then_register(
            np.asarray(aligner.ref_thumbnail),
            np.asarray(aligner.moving_thumbnail),
            max_size=self.task.thumbnail_max_size,
            n_keypoints=self.task.n_keypoints,
            auto_mask=self.task.auto_mask,
            plot_match_result=plot,
        )
        # matrix transforms points from `moving` to `ref`
        affine_matrix = np.vstack([_mx, [0, 0, 1]])

        thumbnail_moving, thumbnail_ref = map(
            skimage.util.img_as_float32,
            [reader_moving.pyramid[-1][0], reader_ref.pyramid[-1][0]],
        )
        tform = skimage.transform.AffineTransform(affine_matrix)

        # warp `moving` image to `ref` image's space
        warped_moving_mask = skimage.transform.warp(
            np.full(thumbnail_moving.shape, fill_value=True, dtype="bool"),
            tform.inverse,
            output_shape=thumbnail_ref.shape,
            cval=False,
        ).astype("uint8")
        _warped_moving = skimage.transform.warp(
            thumbnail_moving,
            tform.inverse,
            output_shape=thumbnail_ref.shape,
            cval=np.median(thumbnail_moving),
        )
        if not np.all(warped_moving_mask == 1):
            dilation_radius = np.floor(
                np.sqrt(warped_moving_mask.sum()) * (np.sqrt(1.1) - 1)
            )
            cv2.dilate(
                warped_moving_mask,
                kernel=skimage.morphology.disk(dilation_radius),
                dst=warped_moving_mask,
            )

        adjust_which, scalar, _ = intensity_config
        mask = warped_moving_mask.astype("bool")
        ref, warped_moving = palom.register_dev.match_img_with_config(
            thumbnail_ref, _warped_moving, mask, mask, adjust_which, scalar, np.array
        )

        return AffineResult(
            ref_img=ref,
            affine_moving_img=warped_moving,
            affine_moving_mask=warped_moving_mask,
            affine_matrix=affine_matrix,
        )


class ElastixAligner(AlignmentStrategy):
    def __init__(self, task: AlignmentTask):
        super().__init__(task)
        self.affine_aligner = AffineAligner(task)

    def align(self, reader_ref, reader_moving, sample_size_factor=3.0, plot=False):
        affine_result = self.affine_aligner.align(reader_ref, reader_moving, plot=plot)

        ref = affine_result.ref_img
        affine_moving = affine_result.affine_moving_img
        affine_moving_mask = affine_result.affine_moving_mask

        # It appears that larger sample size (smaller sample size factor) is
        # useful when the tissue area is small relative to the imaging region.
        # While smaller sample_size works in "common" scenario.
        sample_size = int(np.sqrt(affine_moving_mask.sum()) / sample_size_factor)
        if sample_size >= np.min(ref.shape):
            auto_sample_size = np.min(ref.shape) - 1
            log.info(
                f"Auto reduce sample_size to {auto_sample_size} (was {sample_size})"
            )
            sample_size = auto_sample_size
        n_pxs = int(sample_size**2 * 0.05)

        elastix_moving = np.zeros_like(ref)
        params_tform = None
        error = None
        try:
            elastix_moving, params_tform, _ = elastix_wrapper.run_non_rigid_alignment(
                np.array(ref),
                np.array(affine_moving),
                {
                    "sample_region_size": sample_size,
                    "sample_number_of_pixels": min(n_pxs, 10_000),
                    "grid_size": 60,
                },
                moving_mask=affine_moving_mask,
                ref_mask=affine_moving_mask,
                log=False,
            )
        except RuntimeError as e:
            error = e

        elastix_moving = np.where(
            elastix_moving == 0, np.median(elastix_moving), elastix_moving
        )

        elastix_result = ElastixResult(
            ref_img=ref,
            affine_moving_img=affine_moving,
            elastix_moving_img=elastix_moving,
            transform_params=params_tform,
            error=error,
        )
        return affine_result, elastix_result


# --- ROI Mapper ---
class RoiMapper:
    def __init__(self, conn):
        self.conn = conn

    def fetch_rois(self, image_id):
        return omero_handler.fetch_and_transform_rois(self.conn, image_id, np.eye(3))

    def transform_rois(
        self, rois, reader_from, reader_to, affine_matrix, elastix_params=None
    ):
        tformed_rois = []
        coords = [transform_roi_points.get_roi_points(rr) for rr in rois]
        if not coords:
            return []

        Affine = skimage.transform.AffineTransform

        tform_downscale_from = Affine(
            scale=1 / reader_from.level_downsamples[len(reader_from.pyramid) - 1]
        )
        tform_upscale_to = Affine(
            scale=reader_to.level_downsamples[len(reader_to.pyramid) - 1]
        )
        tform_affine = Affine(matrix=np.linalg.inv(affine_matrix)) + tform_upscale_to

        all_coords = np.vstack(coords)
        # To 'from' thumbnail space
        all_coords = tform_downscale_from(all_coords)

        if elastix_params is not None:
            # apply elastix transformation
            all_coords = elastix_wrapper.map_fixed_points(all_coords, elastix_params)

        # To 'to' full resolution space
        all_coords = tform_affine(all_coords)

        slice_anchors = [0] + np.cumsum([len(cc) for cc in coords]).tolist()
        for rr, (ss, ee) in zip(rois, itertools.pairwise(slice_anchors)):
            tformed_rois.append(
                transform_roi_points.set_roi_points(rr, all_coords[ss:ee].round(1))
            )
        return tformed_rois

    def upload_rois(self, image_id, rois):
        if not rois:
            log.info("No ROIs to upload.")
            return
        omero_handler.post_rois(self.conn, image_id, rois)


# --- QC Plotter ---
class QcPlotter:
    def __init__(self, task: AlignmentTask):
        self.task = task
        self.figures: List[pathlib.Path] = []

    def plot_coarse_alignment(self, reader_from, reader_to):
        import matplotlib.pyplot as plt
        from palom.cli.align_he import set_matplotlib_font, set_subplot_size

        set_matplotlib_font(6)
        task = self.task
        fig, ax = plt.gcf(), plt.gca()

        name1, name2 = self._get_truncated_names(reader_from.path, reader_to.path)

        fig.suptitle(
            f"Coarse Alignment: From {name1} ({task.image_id_from}) To {name2} ({task.image_id_to})",
            fontsize=8,
        )
        ax.set_title(
            f"Channel From: {task.channel_from}, Channel To: {task.channel_to}\n"
            f"Max Pixel Size: {task.max_pixel_size} µm, Keypoints: {task.n_keypoints}\n"
            f"Auto Mask Tissue: {task.auto_mask}, Max Thumbnail Size: {task.thumbnail_max_size}",
            fontsize=6,
        )
        im_h, im_w = ax.images[0].get_array().shape
        set_subplot_size(im_w / 144, im_h / 144, ax=ax)
        ax.set_anchor("N")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        self._add_figure(fig, "coarse")

    def plot_elastix_alignment(
        self, reader_from, reader_to, viz_ref, viz_affine, viz_elastix, skipped
    ):
        import functools

        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from palom.cli.align_he import set_matplotlib_font

        set_matplotlib_font(10)
        failed = np.all(viz_elastix == 0)
        failed_text = ""
        if failed:
            failed_text = " (failed)"
        if skipped:
            failed_text = " (skipped)"
        task = self.task
        Square = functools.partial(mpatches.Rectangle, xy=(0, 0), width=1, height=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.imshow(np.dstack([viz_ref, viz_affine, viz_ref]))
        ax2.imshow(np.dstack([viz_ref, viz_elastix, viz_ref]))
        ax1.set_title("Affine alignment", fontsize=8)
        ax2.set_title("Affine + Elastix alignment" + failed_text, fontsize=8)

        name1, name2 = self._get_truncated_names(reader_from.path, reader_to.path)

        handles = [
            Square(color="magenta", label=f"From: {name1} ({task.image_id_from})"),
            Square(color="lime", label=f"To: {name2} ({task.image_id_to})"),
        ]
        ax1.legend(handles=handles, fontsize=8)
        if failed:
            handles.pop(1)
        ax2.legend(handles=handles, fontsize=8)

        fig.suptitle(
            f"Elastix Alignment: "
            f"From {name1} ({task.image_id_from}, Ch {task.channel_from}) "
            f"To {name2} ({task.image_id_to}, Ch {task.channel_to})",
            fontsize=10,
        )
        self._set_figure_size(fig, ax1.images[0].get_array().shape[:2], 2)
        self._add_figure(fig, "elastix")

    def plot_rois(
        self,
        reader_from,
        reader_to,
        rois_from,
        rois_to,
        viz_from,
        viz_to,
        affine_matrix,
    ):
        import matplotlib.pyplot as plt
        from palom.cli.align_he import set_matplotlib_font

        set_matplotlib_font(10)

        coords_from = [transform_roi_points.get_roi_points(rr) for rr in rois_from]
        coords_to = [transform_roi_points.get_roi_points(rr) for rr in rois_to]

        Affine = skimage.transform.AffineTransform

        tform_downscale_from = Affine(
            scale=1 / reader_from.level_downsamples[len(reader_from.pyramid) - 1]
        )
        tform_downscale_to = Affine(
            scale=1 / reader_to.level_downsamples[len(reader_to.pyramid) - 1]
        )
        tform_affine = tform_downscale_to + Affine(matrix=affine_matrix)

        coords_from = [tform_downscale_from(cc) for cc in coords_from]
        coords_to = [tform_affine(cc) for cc in coords_to]

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        name1, name2 = self._get_truncated_names(reader_from.path, reader_to.path)

        ax1.set_title(f"From {name1} ({self.task.image_id_from})", fontsize=8)
        ax2.set_title(f"To {name2} ({self.task.image_id_to})", fontsize=8)

        ax1.imshow(viz_from, cmap="Greys")
        ax2.imshow(viz_to, cmap="Greys")

        self._draw_rois(ax1, coords_from)
        self._draw_rois(ax2, coords_to)

        fig.suptitle("Mapped ROIs", fontsize=10)
        self._set_figure_size(fig, ax1.images[0].get_array().shape[:2], 2)
        self._add_figure(fig, "roi")

    def save_figures(self):
        if len(self.figures) and (self.task.qc_out_dir is not None):
            import matplotlib.pyplot as plt

            qc_dir = pathlib.Path(self.task.qc_out_dir)
            qc_dir.mkdir(parents=True, exist_ok=True)
            for fig in self.figures:
                fig.savefig(qc_dir / f"{fig.name}.jpg", dpi=144, bbox_inches="tight")
                plt.close(fig)

    def _add_figure(self, fig, suffix):
        pair_label = f"from_{self.task.image_id_from}_to_{self.task.image_id_to}"
        fig.name = f"qc_alignment-{pair_label}-{suffix}"
        self.figures.append(fig)

    @staticmethod
    def _get_truncated_names(name1, name2):
        name1 = str(name1)
        name2 = str(name2)
        if len(name1) > 23:
            name1 = name1[:20] + "..."
        if len(name2) > 23:
            name2 = name2[:20] + "..."
        return name1, name2

    @staticmethod
    def _set_figure_size(fig, shape, num_subplots):
        im_h, im_w = shape
        if im_w < 500:
            im_h *= 500 / im_w
            im_w = 500
        _size_factor = np.divide([im_h, im_w], 2500).max()
        if _size_factor > 1:
            im_h, im_w = np.divide([im_h, im_w], _size_factor)
        fig.set_size_inches(im_w * num_subplots / 144, (im_h + 50) / 144)
        fig.tight_layout(pad=1.5)

    @staticmethod
    def _draw_rois(ax, coords_list):
        for cc in coords_list:
            if len(cc) == 1:
                ax.plot(*cc[0], marker="o", alpha=0.7, color="royalblue")
            elif len(cc) == 2:
                ax.plot(*cc.T, alpha=0.7, color="royalblue")
            elif len(cc) > 2:
                ax.plot(*cc.T, alpha=0.7, color="royalblue")
                ax.fill(*cc.T, alpha=0.3, color="royalblue")


# --- Main Orchestrator ---
class OmeroRoiAligner:
    """Main orchestrator for aligning two OMERO images and mapping ROIs"""

    def __init__(self, conn, task: AlignmentTask, debug=False):
        self.conn = conn
        self.task = task
        self.roi_mapper = RoiMapper(self.conn)
        self.plotter = QcPlotter(self.task)
        self.debug = debug
        if self.debug:
            log.info("Debug mode is ON")

    def execute(self, plot=False):
        handler_from = omero_handler.ImageHandler(self.conn, self.task.image_id_from)
        handler_to = omero_handler.ImageHandler(self.conn, self.task.image_id_to)

        reader_from = palom_wrapper.PalomReaderFactory.create_reader(
            handler=handler_from,
            channel=self.task.channel_from,
            max_pixel_size=self.task.max_pixel_size,
            mask_roi_id=self.task.mask_roi_id_from,
        )
        reader_to = palom_wrapper.PalomReaderFactory.create_reader(
            handler=handler_to,
            channel=self.task.channel_to,
            max_pixel_size=self.task.max_pixel_size,
            mask_roi_id=self.task.mask_roi_id_to,
        )

        if self.task.only_affine:
            strategy = AffineAligner(self.task)
            affine_result = strategy.align(
                reader_ref=reader_from, reader_moving=reader_to, plot=plot
            )
            elastix_result = None
        else:
            strategy = ElastixAligner(self.task)
            affine_result, elastix_result = strategy.align(
                reader_ref=reader_from,
                reader_moving=reader_to,
                sample_size_factor=self.task.sample_size_factor,
                plot=plot,
            )

        if plot:
            int_scalar = 1.0
            if palom.img_util.is_brightfield_img(affine_result.ref_img):
                int_scalar = -1.0

            self.plotter.plot_coarse_alignment(reader_from, reader_to)

            viz_from = self._get_viz_img(int_scalar * affine_result.ref_img)
            viz_to_affine = self._get_viz_img(
                int_scalar * affine_result.affine_moving_img
            )
            viz_to_elastix = np.zeros_like(viz_from)
            if elastix_result:
                viz_to_elastix = self._get_viz_img(
                    int_scalar * elastix_result.elastix_moving_img
                )

            self.plotter.plot_elastix_alignment(
                reader_from,
                reader_to,
                viz_from,
                viz_to_affine,
                viz_to_elastix,
                self.task.only_affine,
            )

        if self.task.map_rois:
            rois = self.roi_mapper.fetch_rois(self.task.image_id_from)
            elastix_params = elastix_result.transform_params if elastix_result else None
            tformed_rois = self.roi_mapper.transform_rois(
                rois,
                reader_from,
                reader_to,
                affine_result.affine_matrix,
                elastix_params,
            )

            if plot:
                self.plotter.plot_rois(
                    reader_from,
                    reader_to,
                    rois,
                    tformed_rois,
                    viz_from,
                    viz_to_affine,
                    affine_result.affine_matrix,
                )

            if not self.task.dry_run:
                self.roi_mapper.upload_rois(self.task.image_id_to, tformed_rois)

        if plot:
            self.plotter.save_figures()

        if self.debug:
            self.handler_from = handler_from
            self.handler_to = handler_to
            self.reader_from = reader_from
            self.reader_to = reader_to
            self.affine_result = affine_result
            self.elastix_result = elastix_result
            if self.task.map_rois:
                self.rois = rois
                self.tformed_rois = tformed_rois
        else:
            if elastix_result is not None and elastix_result.error is not None:
                raise (elastix_result.error)

    def _get_viz_img(self, img):
        in_range = np.percentile(img, [0.1, 99.9])
        return skimage.exposure.adjust_gamma(
            skimage.exposure.rescale_intensity(
                img, in_range=tuple(in_range), out_range="uint8"
            )
            .round()
            .astype("uint8"),
            gain=1.2,
        )
