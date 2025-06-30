import numpy as np
import skimage.util
import skimage.transform
import skimage.exposure
import itk
import pathlib


def map_moving_points(points_xy, param_obj):
    return _map_points(points_xy, param_obj, is_from_moving=True)


def map_fixed_points(points_xy, param_obj):
    return _map_points(points_xy, param_obj, is_from_moving=False)


def _map_points(points_xy, param_obj, is_from_moving=True):
    import scipy.ndimage as ndi

    points = np.asarray(points_xy)
    assert points.shape[1] == 2
    assert points.ndim == 2

    shape = param_obj.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")

    deformation_field = itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), param_obj
    )
    dx, dy = np.moveaxis(deformation_field, 2, 0)

    if is_from_moving:
        inverted_fixed_point = itk.FixedPointInverseDisplacementFieldImageFilter(
            deformation_field,
            NumberOfIterations=10,
            Size=deformation_field.shape[:2][::-1],
        )
        dx, dy = np.moveaxis(inverted_fixed_point, 2, 0)

    my, mx = np.mgrid[: shape[0], : shape[1]].astype("float64")

    mapped_points_xy = np.vstack(
        [
            ndi.map_coordinates(mx + dx, np.fliplr(points).T, mode="nearest"),
            ndi.map_coordinates(my + dy, np.fliplr(points).T, mode="nearest"),
        ]
    ).T
    return mapped_points_xy


def viz_img(img):
    in_range = np.percentile(img, [0.1, 99.9])
    return skimage.exposure.adjust_gamma(
        skimage.exposure.rescale_intensity(
            img, in_range=tuple(in_range), out_range="uint8"
        )
        .round()
        .astype("uint8"),
        gain=2,
    )


def get_default_crc_params(
    grid_size: float = 80.0,
    sample_region_size: float = 300.0,
    sample_number_of_pixels: int = 4_000,
    number_of_iterations: int = 1_000,
):
    parameter_object = itk.ParameterObject.New()
    # deformation
    p = parameter_object.GetDefaultParameterMap("bspline")

    del p["FinalGridSpacingInPhysicalUnits"]

    p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["HowToCombineTransforms"] = ["Compose"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["Transform"] = ["RecursiveBSplineTransform"]

    # metrics: higher weight on the bending energy panelty to reduce distortion
    p["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
    p["Metric0Weight"] = ["1.0"]
    # p["Metric1Weight"] = ["100.0"]
    p["Metric1Weight"] = ["5.0"]

    # these should be pixel-size & image size related
    p["NumberOfResolutions"] = ["4"]
    p["GridSpacingSchedule"] = [f"{2**i}" for i in range(0, 4)[::-1]]
    p["FinalGridSpacingInVoxels"] = [f"{grid_size}", f"{grid_size}"]
    p["NumberOfSamplesForExactGradient"] = [f"{10_000}"]
    p["NumberOfSpatialSamples"] = [f"{sample_number_of_pixels}"]
    # p["NumberOfSpatialSamples"] = [f"{5000 // 2**i}" for i in range(4)[::-1]]
    p["UseRandomSampleRegion"] = ["true"]
    p["SampleRegionSize"] = [f"{sample_region_size}"]
    p["NumberOfHistogramBins"] = ["32"]

    # number if iterations in gradient descent
    p["MaximumNumberOfIterations"] = [f"{number_of_iterations}"]

    # must set to write result image, could be a bug?!
    p["WriteResultImage"] = ["true"]
    # a bug in transformix_jacobian cannot handle using non-default output image
    # format (such as tif, default is nii)
    p["ResultImageFormat"] = ["nii"]

    return p


def run_one_setting(ref_path, moving_path, setting):
    import tifffile

    ref = tifffile.imread(ref_path)
    moving = tifffile.imread(moving_path)

    mref = skimage.exposure.match_histograms(-1.0 * ref, moving).round().astype("uint8")
    return _run_one_setting(mref, moving, setting)


def _run_one_setting(ref, moving, setting, ref_mask=None, moving_mask=None, log=False):
    import tempfile

    if setting is None:
        setting = {}
    elastix_parameter = itk.ParameterObject.New()
    elastix_parameter.AddParameterMap(
        elastix_parameter.GetDefaultParameterMap("rigid", 4)
    )
    elastix_parameter.AddParameterMap(get_default_crc_params(**setting))
    if ref_mask is None:
        ref_mask = np.full(ref.shape, fill_value=1, dtype="uint8")
    if moving_mask is None:
        moving_mask = np.full(moving.shape, fill_value=1, dtype="uint8")

    with tempfile.NamedTemporaryFile(mode="w+") as log_file:
        log_dir = str(pathlib.Path(log_file.name).parent)
        log_filename = pathlib.Path(log_file.name).name
        try:
            warpped_moving, transform_parameter = itk.elastix_registration_method(
                ref,
                moving,
                fixed_mask=itk.image_from_array(ref_mask),
                moving_mask=itk.image_from_array(moving_mask),
                parameter_object=elastix_parameter,
                log_to_console=log,
                log_to_file=True,
                log_file_name=log_filename,
                output_directory=log_dir,
                # log_level=0
            )
        except RuntimeError:
            log_content = log_file.read()
            msg = ""
            if "itk::ExceptionObject" in log_content:
                msg = log_content[log_content.index("itk::ExceptionObject") :]
            raise RuntimeError(
                f"Elastix registration failed the following error\n\n{msg}"
            )
        except Exception as e:
            raise e

    return warpped_moving, transform_parameter, elastix_parameter


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
