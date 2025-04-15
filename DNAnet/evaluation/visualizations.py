import logging
import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from DNAnet.data.data_models import Marker
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.models.prediction import Prediction


LOGGER = logging.getLogger("dnanet")


DNA_CHANNELS = ('blue', 'green', 'black', 'red', 'purple', 'orange')


def _validate_input(hid_images, predictions):
    if (predictions is not None) and (len(hid_images) != len(predictions)):
        raise ValueError(f'Images and predictions do not have the same '
                         f'length. Found {len(hid_images)} images and '
                         f'{len(predictions)} predictions.')


def plot_profile(hid_images: Sequence[HIDImage],
                 predictions: Sequence[Prediction] = None,
                 prediction_as_mask: bool = True,
                 title: bool = True,
                 feature_maps: Sequence[np.ndarray] = None,) -> Optional[plt.Figure]:
    """
    Plot peaks from the DNA profile. If present, called alleles
    are plotted as green mask for each dye.  Optionally, prediction
    results are plotted as a blue  mask (True) or as a line (False).

    :param hid_images: sequence of HID images to plot profile from
    :param predictions: sequence of Prediction for the HID images.
    (which should include an image)
    :param prediction_as_mask: if prediction should be plotted
        as mask or as line
    :param title: whether to include the sample name as a title
    :param feature_maps: optional sequence of 2d ndarray of same size as image,
        to plot on the figure
    """
    _validate_input(hid_images, predictions)

    fig = None
    for enum, image in enumerate(hid_images):
        img = image.data
        if img is None:
            LOGGER.warning(f"No image data found for {image.path}")
            continue
        dyes_maximum_value = img.max(axis=1)
        fig, axs = plt.subplots(len(img), figsize=(20, 20))

        # plot DNA profile
        _plot_profile(axs, img, DNA_CHANNELS)

        # plot called alleles (if present)
        if image.annotation and \
                (image_annotation := image.annotation.image) is not None:
            _plot_segmentation(axs, image_annotation,
                               dyes_maximum_value, 'green')

        # plot predictions (if present)
        if (predictions and
                (image_prediction := predictions[enum].image) is not None):
            if prediction_as_mask:
                _plot_segmentation(axs, (image_prediction > .5).astype(int),
                                   dyes_maximum_value, 'orange')
            else:
                _plot_profile(axs, image_prediction.copy(),
                              'orange', dyes_maximum_value)

        if feature_maps is not None:
            cmap = plt.cm.get_cmap('hsv', len(feature_maps))
            cmap = [cmap(i) for i in range(len(feature_maps))]
            for feature_map, color in zip(feature_maps, cmap):
                _plot_profile(axs, feature_map, color, dyes_maximum_value)

        # plot title
        if title:
            title_string = ""
            if title:
                title_string += f"{image.path.stem.split('/')[-1]}\n"
            fig.suptitle(title_string, fontsize=16)

        plt.show()
    return fig


def _plot_profile(axs,
                  y_vals_per_dye: np.ndarray,
                  color: Any = 'black',
                  max_value_dyes=None,
                  alpha: float = 1) -> None:
    """
    Plot line on each dye of a DNA profile

    :param axs: axes in which each axis represent a dye
    :param y_vals_per_dye: array whose rows contain the values for each dye
    :param color: color for each dye (or single color to use for all dyes).
    color can be str, list of str, or rgb values
    :param max_value_dyes: maximum value for each dye to scale line,
        if provided, the values are rescaled from 0 to this value (per dye)
    :param alpha: alpha used to plot the line
    """
    # Keep zip working
    if max_value_dyes is None:
        max_value_dyes = [None] * len(y_vals_per_dye)

    # Map single color to all dyes
    if isinstance(color, str) or isinstance(color[0], float):
        color = [color] * len(y_vals_per_dye)

    for i, (_color, y_vals, max_value_dye) in \
            enumerate(zip(color, y_vals_per_dye, max_value_dyes)):
        if max_value_dye:
            y_vals *= max_value_dye / max(max(y_vals), 1)
        axs[i].plot(y_vals, c=_color, alpha=alpha)


def _plot_segmentation(axs,
                       image_annotation: np.ndarray,
                       max_value_dyes: Sequence[float],
                       color: str,
                       alpha: float = .5):
    """
    Plot segmentation as background rectangles on each dye, the value `1`
    is considered to indicate the segments

    :param axs: axes in which each axis represent a dye
    :param image_annotation: array in which the rows contain the
        segmentation for each mask (0/1)
    :param max_value_dyes: maximum value for each dye
    :param color: color for rectangles
    :param alpha: alpha use to plot the background
    """
    for i, (dye_segment, max_value_dye) in \
            enumerate(zip(image_annotation, max_value_dyes)):
        axs[i].fill_between(np.arange(len(dye_segment)), 0, max_value_dye,
                            where=dye_segment.flatten() == 1, color=color,
                            alpha=alpha,
                            transform=axs[i].get_xaxis_transform())


def _get_marker_bin(marker):
    left_bin, right_bin = math.inf, -math.inf
    for allele in marker.alleles:
        left_bin = min(left_bin, allele.base_pair - allele.left_bin)
        right_bin = max(right_bin, allele.base_pair + allele.right_bin)

    bins = np.array([left_bin, right_bin])[:, np.newaxis]
    marker_bin = bins + np.array([-1, 1])[:, np.newaxis]

    return marker_bin


def _plot_profile_marker(marker: Marker,
                         image: HIDImage,
                         prediction: Optional[Prediction],
                         ax,
                         zoom_x_values: Optional[Tuple[int, int]]):
    dye_row = marker.dye_row
    marker_bin = _get_marker_bin(marker)
    _slice = np.arange(*tuple(np.argmin(np.abs(image._scaler - marker_bin), axis=1)))
    if zoom_x_values:
        _slice = _slice[slice(*zoom_x_values)]

    marker_img = image.data[dye_row, _slice]
    ax.plot(marker_img, DNA_CHANNELS[dye_row])

    if image.annotation and \
            (image_annotation := image.annotation.image) is not None:
        allele_annotation = image_annotation[dye_row, _slice]
        ax.fill_between(np.arange(len(allele_annotation)), 0, marker_img.max(),
                        where=allele_annotation.flatten() == 1,
                        color='green', alpha=.4, transform=ax.get_xaxis_transform())

    if (prediction and
            (image_prediction := prediction.image) is not None):
        allele_prediction = image_prediction[dye_row, _slice]
        ax.fill_between(np.arange(len(allele_prediction)), 0, marker_img.max(),
                        where=(allele_prediction > .5).astype(int).flatten(),
                        color='orange', alpha=.4, transform=ax.get_xaxis_transform())
    ax.title.set_text(marker.name)


def plot_profile_markers(
    hid_images: Sequence[HIDImage],
    predictions: Sequence[Prediction] = None,
    marker_name: Optional[str] = None,
    zoom_x_values: Optional[Tuple[int, int]] = None
):
    """
    Plot the profile with optional annotations and predictions per marker of for a single
    marker (if `marker_name` is provided). There is the possibility to provide a `zoom_x_values`,
    to manually zoom in on a specific part on the marker.
    """
    _validate_input(hid_images, predictions)

    if predictions is None:
        predictions = [None] * len(hid_images)

    for image, prediction in zip(hid_images, predictions):
        markers = image._panel._panel
        if marker_name:
            fig, axs = plt.subplots(1, 1)
        else:
            fig, axs = plt.subplots(math.ceil(len(markers) / 3), 3, figsize=(20, 40))

        for i, marker in enumerate(markers):
            if marker_name:
                if marker.name == marker_name:
                    _plot_profile_marker(marker, image, prediction, axs, zoom_x_values)
                    break
                else:
                    continue
            else:
                ax = axs[i // 3, i % 3]
                _plot_profile_marker(marker, image, prediction, ax, zoom_x_values)

        fig.suptitle(f'{image.path.name}')
        plt.show()
