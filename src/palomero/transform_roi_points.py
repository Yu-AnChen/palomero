"""Functions for transforming ROI points."""

import dataclasses
import math

import ezomero.rois
import numpy as np
import skimage.transform

from .constants import ELLIPSE_NUM_POINTS


def ellipse_perimeter_ramanujan(a: float, b: float) -> float:
    """
    Calculates the perimeter of an ellipse using Ramanujan's approximation.

    Args:
        a: The length of the semi-major axis.
        b: The length of the semi-minor axis.

    Returns:
        The approximate perimeter of the ellipse.
    """
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def _tform_mx(transform: ezomero.rois.AffineTransform) -> np.ndarray:
    """Converts ezomero.rois' transform property to a 3x3 numpy matrix."""
    return np.array(
        [
            [transform.a00, transform.a01, transform.a02],
            [transform.a10, transform.a11, transform.a12],
            [0, 0, 1],
        ]
    )


def simplify_ellipse(
    x_center: float, y_center: float, x_rad: float, y_rad: float
) -> np.ndarray:
    theta = 0
    t = np.linspace(0, 2 * np.pi, ELLIPSE_NUM_POINTS)
    x = x_rad * np.cos(t) * np.cos(theta) - y_rad * np.sin(t) * np.sin(theta) + x_center
    y = x_rad * np.cos(t) * np.sin(theta) + y_rad * np.sin(t) * np.cos(theta) + y_center
    return np.array([x, y]).T


def get_roi_points(roi: ezomero.rois.ezShape) -> np.ndarray:
    roi_classes = (
        ezomero.rois.Ellipse,
        ezomero.rois.Label,
        ezomero.rois.Line,
        ezomero.rois.Point,
        ezomero.rois.Polygon,
        ezomero.rois.Polyline,
        ezomero.rois.Rectangle,
    )
    if not isinstance(roi, roi_classes):
        raise TypeError(f"Expected an ezomero ROI shape, got {type(roi).__name__}")

    if isinstance(roi, ezomero.rois.Ellipse):
        coords = simplify_ellipse(roi.x, roi.y, roi.x_rad, roi.y_rad)
    elif isinstance(roi, (ezomero.rois.Label, ezomero.rois.Point)):
        coords = np.array([[roi.x, roi.y]])
    elif isinstance(roi, ezomero.rois.Line):
        coords = np.array([[roi.x1, roi.y1], [roi.x2, roi.y2]])
    elif isinstance(roi, (ezomero.rois.Polygon, ezomero.rois.Polyline)):
        coords = np.array(roi.points)
    elif isinstance(roi, ezomero.rois.Rectangle):
        coords = np.array(
            [
                [roi.x, roi.y],
                [roi.x + roi.width, roi.y],
                [roi.x + roi.width, roi.y + roi.height],
                [roi.x, roi.y + roi.height],
            ]
        )
    if roi.transform is not None:
        coords = skimage.transform.AffineTransform(matrix=_tform_mx(roi.transform))(
            coords
        )
    return coords


def set_roi_points(
    roi: ezomero.rois.ezShape, coords: np.ndarray
) -> ezomero.rois.ezShape:
    roi_classes = (
        ezomero.rois.Ellipse,
        ezomero.rois.Label,
        ezomero.rois.Line,
        ezomero.rois.Point,
        ezomero.rois.Polygon,
        ezomero.rois.Polyline,
        ezomero.rois.Rectangle,
    )
    if not isinstance(roi, roi_classes):
        raise TypeError(f"Expected an ezomero ROI shape, got {type(roi).__name__}")

    common_keys = set.intersection(
        *[{ff.name for ff in dataclasses.fields(cc)} for cc in roi_classes]
    )
    props = {kk: getattr(roi, kk, None) for kk in common_keys}
    props.update({"transform": None})

    if isinstance(roi, (ezomero.rois.Ellipse, ezomero.rois.Rectangle)):
        out_roi = ezomero.rois.Polygon(points=coords, **props)
    elif isinstance(roi, (ezomero.rois.Label, ezomero.rois.Point)):
        [(x, y)] = coords
        out_roi = dataclasses.replace(roi, transform=None, x=x, y=y)
    elif isinstance(roi, ezomero.rois.Line):
        [(x1, y1), (x2, y2)] = coords
        out_roi = dataclasses.replace(roi, transform=None, x1=x1, y1=y1, x2=x2, y2=y2)
    elif isinstance(roi, (ezomero.rois.Polygon, ezomero.rois.Polyline)):
        out_roi = dataclasses.replace(roi, transform=None, points=coords)
    return out_roi
