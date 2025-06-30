"""Main alignment orchestrator class."""

import itertools
import logging
import pathlib

import cv2
import numpy as np
import palom
import skimage.morphology
import skimage.transform
import skimage.util

from . import elastix_wrapper, palom_wrapper
from .. import omero_handler
from ..models import AlignmentTask
from .. import transform_roi_points

log = logging.getLogger(__name__)


class OmeroRoiAligner:
    def __init__(self, conn, task: AlignmentTask):
        self.conn = conn
        self.task = task
        self.params_tform = None
        self.roi = None
        self.tformed_rois = None
        self.figures = []

    def execute(self, plot=False):
        self.get_readers()
        self.affine_align_readers(plot=plot)
        if plot:
            self.annotate_coarse_alignment_plot()
        if not self.task.affine_only:
            self.elastix_align_readers()
            if plot:
                self.plot_alignment()
        if self.task.map_rois:
            self.map_rois()
            if plot:
                self.plot_rois()
            if not self.task.dry_run:
                self.upload_transformed_rois()
        if len(self.figures) & (self.task.qc_out_dir is not None):
            import matplotlib.pyplot as plt

            qc_dir = pathlib.Path(self.task.qc_out_dir)
            qc_dir.mkdir(parents=True, exist_ok=True)
            for ff in self.figures:
                ff.savefig(qc_dir / f"{ff.name}.jpg", dpi=144, bbox_inches="tight")
                plt.close(ff)

    def get_readers(self):
        handler_to = omero_handler.ImageHandler(self.conn, self.task.image_id_to)
        handler_from = omero_handler.ImageHandler(self.conn, self.task.image_id_from)

        self.reader1 = palom_wrapper.PalomReaderFactory.create_reader(
            handler_from, self.task.channel_from, self.task.max_pixel_size
        )
        self.reader2 = palom_wrapper.PalomReaderFactory.create_reader(
            handler_to, self.task.channel_to, self.task.max_pixel_size
        )

    def affine_align_readers(self, plot):
        self.aligner = palom.align.get_aligner(self.reader1, self.reader2)
        _mx, intensity_config = palom_wrapper.search_then_register(
            np.asarray(self.aligner.ref_thumbnail),
            np.asarray(self.aligner.moving_thumbnail),
            max_size=self.task.thumbnail_max_size,
            n_keypoints=self.task.n_keypoints,
            auto_mask=self.task.auto_mask,
            plot_match_result=plot,
        )
        self.aligner.coarse_affine_matrix = np.vstack([_mx, [0, 0, 1]])
        self.intensity_config = intensity_config

    def elastix_align_readers(self):
        img1, img2 = map(
            skimage.util.img_as_float32,
            [self.reader1.pyramid[-1][0], self.reader2.pyramid[-1][0]],
        )
        tform = skimage.transform.AffineTransform(self.aligner.coarse_affine_matrix)

        moving_mask = skimage.transform.warp(
            np.full(img2.shape, fill_value=True, dtype="bool"),
            tform.inverse,
            output_shape=img1.shape,
            cval=False,
        ).astype("uint8")
        moving = skimage.transform.warp(
            img2,
            tform.inverse,
            output_shape=img1.shape,
            cval=np.median(img2),
        )
        if not np.all(moving_mask == 1):
            dilation_radius = np.floor(np.sqrt(moving_mask.sum()) * (np.sqrt(1.1) - 1))
            cv2.dilate(
                moving_mask,
                kernel=skimage.morphology.disk(dilation_radius),
                dst=moving_mask,
            )

        adjust_which, scalar, _ = self.intensity_config
        mask = moving_mask.astype("bool")
        ref, mmoving = palom.register_dev.match_img_with_config(
            img1, moving, mask, mask, adjust_which, scalar, np.array
        )

        sample_size = int(np.sqrt(moving_mask.sum()) / 3.0)
        n_pxs = int(sample_size**2 * 0.05)

        img = np.zeros_like(ref)
        params_tform, params_config = None, None
        try:
            img, params_tform, params_config = elastix_wrapper.run_non_rigid_alignment(
                np.array(ref),
                np.array(mmoving),
                {
                    "sample_region_size": sample_size,
                    "sample_number_of_pixels": min(n_pxs, 10_000),
                    "grid_size": 60,
                },
                moving_mask=moving_mask,
                ref_mask=moving_mask,
                log=False,
            )
        except RuntimeError as e:
            log.error(f"Elastix registration failed: {e}")

        self.params_tform = params_tform

        img = np.where(img == 0, np.median(img), img)
        iinv = -1.0 if palom.img_util.is_brightfield_img(ref) else 1

        self.viz_ref = elastix_wrapper.viz_img(iinv * ref)
        self.viz_mmoving = elastix_wrapper.viz_img(iinv * mmoving)
        self.viz_img = elastix_wrapper.viz_img(iinv * img)

    def map_rois(self):
        if self.roi is None:
            self.rois = omero_handler.fetch_and_transform_rois(
                self.conn, self.task.image_id_from, np.eye(3)
            )
        self.tformed_rois = []
        coords = [transform_roi_points.get_roi_points(rr) for rr in self.rois]

        Affine = skimage.transform.AffineTransform

        tform_before = Affine(
            scale=1 / self.reader1.level_downsamples[len(self.reader1.pyramid) - 1]
        )
        tform_after = Affine(
            matrix=np.linalg.inv(self.aligner.coarse_affine_matrix)
        ) + Affine(scale=self.reader2.level_downsamples[len(self.reader2.pyramid) - 1])
        slice_anchors = [0] + np.cumsum([len(cc) for cc in coords]).tolist()
        if self.params_tform is None:
            mapped_coords = tform_after(tform_before(np.vstack(coords)))
        else:
            mapped_coords = tform_after(
                elastix_wrapper.map_fixed_points(tform_before(np.vstack(coords)), self.params_tform)
            )
        for rr, (ss, ee) in zip(self.rois, itertools.pairwise(slice_anchors)):
            self.tformed_rois.append(transform_roi_points.set_roi_points(rr, mapped_coords[ss:ee].round(1)))

    def upload_transformed_rois(self):
        if self.tformed_rois is None:
            print("Alignment task has not been run. No ROIs to upload.")
            return
        omero_handler.post_rois(self.conn, self.task.image_id_to, self.tformed_rois)

    def annotate_coarse_alignment_plot(self):
        import matplotlib.pyplot as plt
        from palom.cli.align_he import set_matplotlib_font, set_subplot_size

        set_matplotlib_font(6)

        task = self.task
        fig, ax = plt.gcf(), plt.gca()

        name1 = self.reader1.path
        name2 = self.reader2.path
        if len(name1) > 23:
            name1 = name1[:20] + "..."
        if len(name2) > 23:
            name2 = name2[:20] + "..."

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

        pair_label = f"from_{task.image_id_from}_to_{task.image_id_to}"
        qc_filename_prefix = f"qc_alignment-{pair_label}-coarse"
        fig.name = qc_filename_prefix

        self.figures.append(fig)

    def plot_alignment(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import functools
        from palom.cli.align_he import set_matplotlib_font

        set_matplotlib_font(10)

        task = self.task

        Square = functools.partial(mpatches.Rectangle, xy=(0, 0), width=1, height=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.imshow(np.dstack([self.viz_ref, self.viz_mmoving, self.viz_ref]))
        ax2.imshow(np.dstack([self.viz_ref, self.viz_img, self.viz_ref]))
        ax1.set_title("Affine alignment", fontsize=8)
        ax2.set_title("Affine + Elastix alignment", fontsize=8)
        name1 = self.reader1.path
        name2 = self.reader2.path
        if len(name1) > 23:
            name1 = name1[:20] + "..."
        if len(name2) > 23:
            name2 = name2[:20] + "..."
        handles = [
            Square(
                color="lime",
                label=f"Img 1: {name2} ({task.image_id_to})",
            ),
            Square(
                color="magenta",
                label=f"Img 2: {name1} ({task.image_id_from})",
            ),
        ]
        ax1.legend(handles=handles, fontsize=8)
        ax2.legend(handles=handles, fontsize=8)

        fig.suptitle(
            f"Elastix Alignment: "
            f"From {name1} ({task.image_id_from}, Ch {task.channel_from}) "
            f"To {name2} ({task.image_id_to}, Ch {task.channel_to})",
            fontsize=10,
        )
        im_h, im_w = ax1.images[0].get_array().shape[:2]
        if im_w < 500:
            im_h *= 500 / im_w
            im_w = 500
        _size_factor = np.divide([im_h, im_w], 2500).max()
        if _size_factor > 1:
            im_h, im_w = np.divide([im_h, im_w], _size_factor)
        fig.set_size_inches(im_w * 2 / 144, (im_h + 50) / 144)
        fig.tight_layout(pad=1.5)

        pair_label = f"from_{task.image_id_from}_to_{task.image_id_to}"
        qc_filename_prefix = f"qc_alignment-{pair_label}-elastix"
        fig.name = qc_filename_prefix

        self.figures.append(fig)

    def plot_rois(self):
        import matplotlib.pyplot as plt
        from palom.cli.align_he import set_matplotlib_font

        set_matplotlib_font(10)
        Affine = skimage.transform.AffineTransform
        tform_before = Affine(
            scale=1 / self.reader1.level_downsamples[len(self.reader1.pyramid) - 1]
        )
        coords = [tform_before(transform_roi_points.get_roi_points(rr)) for rr in self.rois]

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

        name1 = self.reader1.path
        name2 = self.reader2.path
        if len(name1) > 23:
            name1 = name1[:20] + "..."
        if len(name2) > 23:
            name2 = name2[:20] + "..."

        ax1.set_title(f"From {name1} ({self.task.image_id_from})", fontsize=8)
        ax2.set_title(f"To {name2} ({self.task.image_id_to})", fontsize=8)

        ax1.imshow(self.viz_ref, cmap="Greys")
        if np.any(self.viz_img):
            ax2.imshow(self.viz_img, cmap="Greys")
        else:
            ax2.imshow(self.viz_mmoving, cmap="Greys")

        for cc in coords:
            if len(cc) == 1:
                ax1.plot(*cc[0], marker="o", alpha=0.7, color="royalblue")
                ax2.plot(*cc[0], marker="o", alpha=0.7, color="royalblue")
            elif len(cc) == 2:
                ax1.plot(*cc.T, alpha=0.7, color="royalblue")
                ax2.plot(*cc.T, alpha=0.7, color="royalblue")
            elif len(cc) > 2:
                for ax in (ax1, ax2):
                    ax.plot(*cc.T, alpha=0.7, color="royalblue")
                    ax.fill(*cc.T, alpha=0.3, color="royalblue")
            else:
                continue

        fig.suptitle("Mapped ROIs", fontsize=10)
        im_h, im_w = ax1.images[0].get_array().shape[:2]
        if im_w < 500:
            im_h *= 500 / im_w
            im_w = 500
        _size_factor = np.divide([im_h, im_w], 2500).max()
        if _size_factor > 1:
            im_h, im_w = np.divide([im_h, im_w], _size_factor)
        fig.set_size_inches(im_w * 2 / 144, (im_h + 50) / 144)
        fig.tight_layout(pad=1.5)

        pair_label = f"from_{self.task.image_id_from}_to_{self.task.image_id_to}"
        qc_filename_prefix = f"qc_alignment-{pair_label}-roi"
        fig.name = qc_filename_prefix

        self.figures.append(fig)
