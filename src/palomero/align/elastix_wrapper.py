"""Wraps ITK Elastix for non-rigid alignment."""

import itk
import numpy as np
import skimage.transform
import skimage.util


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


def get_default_crc_params(
    grid_size: float = 80.0,
    sample_region_size: float = 300.0,
    sample_number_of_pixels: int = 4_000,
    number_of_iterations: int = 1_000,
):
    parameter_object = itk.ParameterObject.New()
    p = parameter_object.GetDefaultParameterMap("bspline")

    del p["FinalGridSpacingInPhysicalUnits"]

    p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["HowToCombineTransforms"] = ["Compose"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["Transform"] = ["RecursiveBSplineTransform"]
    p["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
    p["Metric0Weight"] = ["1.0"]
    p["Metric1Weight"] = ["5.0"]
    p["NumberOfResolutions"] = ["4"]
    p["GridSpacingSchedule"] = [f"{2**i}" for i in range(0, 4)[::-1]]
    p["FinalGridSpacingInVoxels"] = [f"{grid_size}", f"{grid_size}"]
    p["NumberOfSamplesForExactGradient"] = [f"{10_000}"]
    p["NumberOfSpatialSamples"] = [f"{sample_number_of_pixels}"]
    p["UseRandomSampleRegion"] = ["true"]
    p["SampleRegionSize"] = [f"{sample_region_size}"]
    p["NumberOfHistogramBins"] = ["32"]
    p["MaximumNumberOfIterations"] = [f"{number_of_iterations}"]
    p["WriteResultImage"] = ["true"]
    p["ResultImageFormat"] = ["nii"]

    return p


def run_non_rigid_alignment(
    ref, moving, setting, ref_mask=None, moving_mask=None, log=False
):
    import pathlib
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

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as log_file:
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
            )
        except RuntimeError:
            log_content = log_file.read()
            msg = ""
            if "itk::ExceptionObject" in log_content:
                msg = log_content[log_content.index("itk::ExceptionObject") :]
            raise RuntimeError(
                f"Elastix registration failed with the following error\n\n{msg}"
            )
        except Exception as e:
            raise e

    return warpped_moving, transform_parameter, elastix_parameter
