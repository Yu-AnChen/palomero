import pathlib

import cv2
import dask.array as da
import dask.diagnostics
import dask_image.ndinterp
import numpy as np
import palom
import skimage.transform
import zarr
from numcodecs import Zstd
from palom.cli import align_he
from palom.reader import DaPyramidChannelReader

from palomero.align.aligner import ElastixAligner, OmeroRoiAligner, QcPlotter
from palomero.models import AlignmentTask


class LocalPalomeroAligner:
    """
    Orchestrates the alignment of two local images, generates QC plots,
    and warps the moving image, saving it as an OME-TIFF.
    """

    def __init__(
        self,
        path_from: str,
        path_to: str,
        out_path: str,
        task: AlignmentTask,
        temp_zarr_store_dir: str,
    ):
        self.path_from = path_from
        self.path_to = path_to
        self.out_path = out_path
        self.task = task
        self.temp_zarr_store_dir = temp_zarr_store_dir
        self.base_reader_from = None
        self.base_reader_to = None
        self.reader_from = None
        self.reader_to = None
        self.affine_result = None
        self.elastix_result = None

        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assert "".join(out_path.suffixes[-2:]) in (".ome.tif", ".ome.tiff")

    def run(self):
        """Executes the entire alignment and warping workflow."""
        print("1. Reading and preparing images...")
        self._prepare_readers()

        print("\n2. Performing alignment...")
        self._align_images()

        print("\n3. Generating QC plots...")
        self._generate_qc_plots()

        print("\n4. Warping moving image...")
        self._warp_and_save_image()

        print(f"\nWorkflow finished. Warped image saved to '{self.out_path}'")

    def _prepare_readers(self):
        """Creates and configures palom readers for local images."""

        self.base_reader_from = align_he.get_reader(self.path_from)(self.path_from)
        self.base_reader_to = align_he.get_reader(self.path_to)(self.path_to)

        self.reader_from = self._create_dask_reader(
            self.base_reader_from,
            channel=self.task.channel_from,
            max_pixel_size=self.task.max_pixel_size,
        )
        self.reader_to = self._create_dask_reader(
            self.base_reader_to,
            channel=self.task.channel_to,
            max_pixel_size=self.task.max_pixel_size,
        )

    def _create_dask_reader(
        self, base_reader, channel: int, max_pixel_size: float
    ) -> DaPyramidChannelReader:
        """Creates a configured DaPyramidChannelReader from a base palom reader."""
        level_pixel_sizes = [
            base_reader.level_downsamples[ii] * base_reader.pixel_size
            for ii in range(len(base_reader.pyramid))
        ]
        valid_levels = np.where(np.array(level_pixel_sizes) <= max_pixel_size)[0]
        level = int(np.max(valid_levels)) if len(valid_levels) > 0 else 0

        img = np.array(base_reader.pyramid[level][channel])

        level_shapes = [pp.shape[1:3] for pp in base_reader.pyramid[: level + 1]]
        pyramid = [
            da.zeros((1, *ss), chunks=1024, dtype=img.dtype, name=False)
            for ss in level_shapes
        ]
        pyramid[level] = da.from_array(img[np.newaxis], chunks=(1, 1024, 1024))

        tip_pixel_size = level_pixel_sizes[level]
        while tip_pixel_size < max_pixel_size / 2:
            tip_pixel_size *= 2
            img_to_downsample = np.array(pyramid[-1][0])
            downsampled_img = palom.img_util.cv2_downscale_local_mean(
                img_to_downsample, 2
            )
            pyramid.append(
                da.from_array(downsampled_img[np.newaxis], chunks=(1, 1024, 1024))
            )

        dask_reader = DaPyramidChannelReader(pyramid, channel_axis=0)
        dask_reader.pixel_size = base_reader.pixel_size
        dask_reader.path = base_reader.path
        if level == 0:
            dask_reader.pyramid = dask_reader.pyramid[:1]
        return dask_reader

    def _align_images(self):
        """Runs the core alignment algorithm."""
        strategy = ElastixAligner(self.task)
        self.affine_result, self.elastix_result = strategy.align(
            reader_ref=self.reader_from,
            reader_moving=self.reader_to,
            plot=True,
        )

    def _generate_qc_plots(self):
        """Generates and saves QC plots for the alignment."""
        plotter = QcPlotter(self.task)
        # The OmeroRoiAligner's _get_viz_img is a convenient helper
        to_viz_img = OmeroRoiAligner(None, None)._get_viz_img

        int_scalar = 1.0
        if palom.img_util.is_brightfield_img(self.affine_result.ref_img):
            int_scalar = -1.0

        viz_from = to_viz_img(int_scalar * self.affine_result.ref_img)
        viz_to_affine = to_viz_img(int_scalar * self.affine_result.affine_moving_img)
        viz_to_elastix = np.zeros_like(viz_from)
        if self.elastix_result:
            viz_to_elastix = to_viz_img(
                int_scalar * self.elastix_result.elastix_moving_img
            )

        plotter.plot_coarse_alignment(self.reader_from, self.reader_to)
        plotter.plot_elastix_alignment(
            self.reader_from,
            self.reader_to,
            viz_from,
            viz_to_affine,
            viz_to_elastix,
            self.task.only_affine,
        )

        plotter.save_figures()
        print(f"QC plots saved in '{self.task.qc_out_dir}'.")

    def _warp_and_save_image(self, _level=0):
        """Warps the moving image and saves it as a pyramidal OME-TIFF."""

        r1, r2 = self.base_reader_from, self.base_reader_to
        LEVEL = _level  # Warp the highest resolution
        out_level1, out_level2 = palom.align_multi_res.match_levels(r1, r2)[LEVEL]

        affine_level2 = max(self.reader_to.level_downsamples.keys())
        affine_level1 = max(self.reader_from.level_downsamples.keys())
        # FIXME: temporary rounding
        d_moving = np.round(
            self.reader_to.level_downsamples[affine_level2]
            / r2.level_downsamples[out_level2]
        )
        d_ref = np.round(
            self.reader_from.level_downsamples[affine_level1]
            / r1.level_downsamples[out_level1]
        )

        Affine = skimage.transform.AffineTransform

        mx = self.affine_result.affine_matrix
        _dform = elastix_param_to_dform(self.elastix_result.transform_params)

        tform = Affine(scale=1 / d_moving) + Affine(matrix=mx) + Affine(scale=d_ref)

        dform = d_ref * _dform
        out_shape = r1.pyramid[out_level1].shape[1:3]

        dydx = da.array(
            [
                dask_image.ndinterp.affine_transform(
                    dd,
                    matrix=np.linalg.inv(Affine(scale=d_ref).params),
                    output_shape=out_shape,
                    output_chunks=(1024, 1024),
                )
                for dd in dform[::-1]
            ]
        )
        gygx = da.indices(out_shape, dtype="float", chunks=(1024, 1024))
        gygx += dydx

        gygx = gygx.rechunk((2, 1024, 1024))

        # the chunk size (256, 256, 3) isn't ideal to be loaded with dask;
        # hard-code the reading and axis swap
        temp_zarr_store_dir = self.temp_zarr_store_dir
        _moving = r2.pyramid[out_level2]
        chunks = np.ceil(np.divide(2048, _moving.chunksize[1:3])) * np.array(
            _moving.chunksize[1:3]
        )
        store = None
        if temp_zarr_store_dir is not None:
            store = zarr.TempStore(dir=temp_zarr_store_dir)
        moving = zarr.group(store=store, overwrite=True)
        for idx, channel in enumerate(_moving):
            moving[idx] = zarr.empty(
                channel.shape,
                chunks=chunks.astype("int"),
                dtype=_moving.dtype,
                compressor=Zstd(),
            )
            with dask.diagnostics.ProgressBar():
                channel.to_zarr(moving[idx])

        pre_filter_sigma = 1 if tform.scale.min() < 1 else 0
        mosaics = []
        for channel in moving.values():
            sr, sc = np.ceil(np.divide(channel.shape, 1000)).astype("int")
            cval = np.percentile(np.array(channel[::sr, ::sc]), 75).item()
            warped_moving = gygx.map_blocks(
                _wrap_cv2_large_proper,
                channel,
                mx=np.linalg.inv(tform.params),
                cval=cval,
                sigma=pre_filter_sigma,
                module="skimage",
                dtype=channel.dtype,
                drop_axis=0,
            )

            mosaics.append(warped_moving)
        palom.pyramid.write_pyramid(
            mosaics,
            output_path=self.out_path,
            pixel_size=r1.pixel_size * r1.level_downsamples[out_level1],
        )


def _wrap_cv2_large_proper(mapping_yx, img, mx, cval, sigma=0, module="cv2"):
    assert module in ["cv2", "skimage"]
    assert mx.shape == (3, 3)
    assert mapping_yx.ndim == 3
    assert mapping_yx.shape[0] == 2, mapping_yx.shape

    _, H, W = mapping_yx.shape

    mapping_yx = np.array(mapping_yx)
    if not np.all(mx == 0):
        # matrix multiplication is faster using cv2
        # tform = skimage.transform.AffineTransform(mx)
        # mapping_yx = tform(mapping_yx.reshape(2, -1)[::-1].T).T[::-1].reshape(2, H, W)

        mapping_yx = (
            cv2.transform(mapping_yx.reshape(2, -1)[::-1].T[:, np.newaxis], mx[:2])
            .squeeze()
            .T[::-1]
            .reshape(2, H, W)
        )
    # add extra pixel for linear interpolation
    rmin, cmin = np.floor(mapping_yx.min(axis=(1, 2))).astype("int") - 1
    rmax, cmax = np.ceil(mapping_yx.max(axis=(1, 2))).astype("int") + 1

    if np.any(np.asarray([rmax, cmax]) <= 0):
        return np.full((H, W), fill_value=cval, dtype=img.dtype)

    rmin, cmin = np.clip([rmin, cmin], 0, None)
    rmax, cmax = np.clip([rmax, cmax], None, img.shape)

    mapping_yx -= np.reshape([rmin, cmin], (2, 1, 1))

    # cast mapping down to 32-bit float for speed and compatibility
    mapping_yx = mapping_yx.astype("float32")

    crop_img = np.array(img[rmin:rmax, cmin:cmax])

    if 0 in crop_img.shape:
        return np.full((H, W), fill_value=cval, dtype=img.dtype)

    if sigma != 0:
        pad = sigma * 4
        pad_rmin, pad_cmin = np.clip(np.subtract([rmin, cmin], pad), 0, None)
        pad_rmax, pad_cmax = np.clip(np.add([rmax, cmax], pad), None, img.shape)
        _crop_img = np.array(img[pad_rmin:pad_rmax, pad_cmin:pad_cmax])
        border_mode = cv2.BORDER_REPLICATE
        _crop_img = cv2.GaussianBlur(_crop_img, (0, 0), sigma, borderType=border_mode)
        crop_img = _crop_img[
            rmin - pad_rmin : rmin - pad_rmin + crop_img.shape[0],
            cmin - pad_cmin : cmin - pad_cmin + crop_img.shape[1],
        ]

    if 0 in img.shape:
        return np.full((H, W), fill_value=cval, dtype=img.dtype)
    if module == "cv2":
        return cv2.remap(
            crop_img, mapping_yx[1], mapping_yx[0], cv2.INTER_LINEAR, borderValue=cval
        )
    return skimage.transform.warp(
        crop_img, mapping_yx, preserve_range=True, cval=cval
    ).astype(crop_img.dtype)


def elastix_param_to_dform(param_obj):
    import itk

    shape = param_obj.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")

    deformation_field = itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), param_obj
    )
    return np.moveaxis(deformation_field, 2, 0)


def parse_lsp_id(name):
    import re

    name = str(name)
    match = re.search(r"LSP\d{5}", name)
    lsp_id = None
    if match:
        lsp_id = match.group(0)
    return lsp_id
