import logging
import math
from typing import Any, Optional, Sequence, Tuple, List, Dict, Type

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.patches import Rectangle


from DNAnet.constants import LABEL_CATEGORIES
from DNAnet.data.data_models import Marker
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.data.utils import CATEGORIES, DNA_CHANNELS, RESCALE_SIZE, full_annotation_array_to_list
from DNAnet.evaluation.interactivity import Interactivity
from DNAnet.evaluation.utils import get_peaks
from DNAnet.models.base import Model
from DNAnet.models.prediction import Prediction
from DNAnet.utils import _get_marker_bin, get_allele_bins
from config_io import load_model

LOGGER = logging.getLogger("dnanet")

SLOPE = (475.0 - 65.0) / 4096.0
OFFSET = 65.0


def _validate_input(hid_images, predictions):
    if (predictions is not None) and (len(hid_images) != len(predictions)):
        raise ValueError(f'Images and predictions do not have the same '
                         f'length. Found {len(hid_images)} images and '
                         f'{len(predictions)} predictions.')


def plot_profile(hid_images: Sequence[HIDImage],
                 predictions: Sequence[Prediction] = None,
                 prediction_as_mask: bool = True,
                 title: bool = True,
                 feature_maps: Sequence[np.ndarray] = None,
                 min_rfu_peak_detection: Optional[int] = 200,
                 plot_allele_bins: bool = False,
                 interactive: Optional[Type[Interactivity]] = None,
                 spans_by_profile: Optional[Dict[str, List[Dict[str, object]]]] = None
                 ) -> Optional[plt.Figure]:
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
    :param plot_allele_bins: whether to plot the allele bins.
    :param interactive: When given plot is given interactive functionalities from Interactivity
    :param min_rfu_peak_detection: if no spans are provided, we can automatically detect peaks above a certain RFU threshold. If None, no peaks are detected.
    :param spans_by_profile: Optionally provide pre-made annotations, for one or more profiles
    """
    _validate_input(hid_images, predictions)

    fig = None
    for enum, image in enumerate(hid_images):
        img = image.data
        if img is None:
            LOGGER.warning(f"No image data found for {image.path}")
            continue
        dyes_maximum_value = img.max(axis=1)
        fig, axs = plt.subplots(len(img), figsize=(20, 20), sharex=True)

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

        if plot_allele_bins:
            marker_ys = []
            for ax in axs:
                # Add extra space for the annotations
                ymin, ymax = ax.get_ylim()
                extra_space = (ymax - ymin) * 0.1
                marker_ys.append(ymin - extra_space / 2)
                ax.set_ylim(ymin - extra_space, ymax)
                if not interactive:
                    # dont override y-ticks in dynamic view (eg labelling) so zooming updates them
                    ax.set_yticks([tick for tick in ax.get_yticks() if 0 <= tick])

            allele_bins = get_allele_bins(image)
            for i_dye, allele_bin in allele_bins:
                axs[i_dye].plot(allele_bin,[marker_ys[i_dye], marker_ys[i_dye]])


        if feature_maps is not None:
            cmap = plt.cm.get_cmap('hsv', len(feature_maps))
            cmap = [cmap(i) for i in range(len(feature_maps))]
            for feature_map, color in zip(feature_maps, cmap):
                _plot_profile(axs, feature_map, color, dyes_maximum_value)


        profile_name = image.path.stem.split('/')[-1]
        spans = None
        users = None
        if spans_by_profile:
            spans = spans_by_profile[profile_name]
            users = list(set([span['annotator'] for span in spans]))
            plt.suptitle(' '.join(users))
            # map the annotation names to colours
            for span in spans:
                # add the right axis and artist:
                i_dye = DNA_CHANNELS.index(span['dye'])
                span['ax'] = axs[i_dye]
                if len(users) == 1:
                    span['artist'] = axs[i_dye].axvspan(span['x0'], span['x1'],
                                       color=span['color'],
                                       alpha=span['alpha'])
                else:
                    # if multiple annotators, plot their annotations on top of each other
                    y_min, y_max = axs[i_dye].get_ylim()
                    y_range = y_max - y_min
                    rect_height = y_range / len(users)
                    # Plot n rectangles stacked vertically
                    i = users.index(span['annotator'])
                    y_bottom = y_min + i * rect_height
                    rect = Rectangle((span['x0'], y_bottom),
                                     width=span['x1'] - span['x0'],
                                     height=rect_height,
                                     facecolor=span['color'],
                                     # edgecolor=span['color'],
                                     alpha=0.7)
                    span['artist'] = axs[i_dye].add_patch(rect)
            if spans:
                print(f'read {len(spans)} annotations from file for {profile_name}')
        title_string = ""

        if not spans and min_rfu_peak_detection is not None:
            if img.shape[1]==RESCALE_SIZE:
                # if we have a 'regular' sized profile, we look for peaks
                spans = _add_initial_spans(
                    axs=axs, peak_ranges=get_peaks(profile=img, min_rfu=min_rfu_peak_detection)
                )
                print(f'No previous annotations; automatically detected {len(spans)} peaks for'
                      f'{profile_name} and threshold {min_rfu_peak_detection}')
            else:
                # if we could not parse ILS, we have many more pixels (raw data).
                # We show this profile to ask for annotations on the ILS
                title_string = 'annotate ILS: '

        # if not spans:
        #     ## TODO: generate spans with a pretrained model
        #     model = load_model('unet.yaml')
        #     model.load('resources/hackathon/current_best')
        #     if img.shape[1] == RESCALE_SIZE:
        #         # if we have a 'regular' sized profile, we look for peaks
        #         spans = _predict_initial_spans(
        #             model, image, axs=axs
        #         )
        #         print(f'No previous annotations; automatically detected {len(spans)} peaks for '
        #               f'{profile_name}')
        #     else:
        #         # if we could not parse ILS, we have many more pixels (raw data).
        #         # We show this profile to ask for annotations on the ILS
        #         title_string = 'annotate ILS: '


        if title:
            title_string += f"{profile_name}\n"
            if users:
                # plot the annotators if they exist
                title_string+=' '.join(users)
            fig.suptitle(title_string, fontsize=16)

        if interactive is not None:
            interactive_instance = interactive(fig, axs, spans)
            interactive_instance.activate_interactivity()

        #  todo: make dynamic axis labels? When you zoom, you lose the X-axis labels.to
        major_bp = np.arange(0, 500, 25)  # 65, 476: 65, 90, 115, ..., 465
        major_scan = bp_to_scan(major_bp)
        ax.xaxis.set_major_locator(FixedLocator(major_scan))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{scan_to_bp(x):.0f}"))

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
        axs[i].set_ylim(0, max_value_dye)


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


def _predict_initial_spans(model: Model, image: HIDImage, axs) -> List[Dict[str, object]]:
    """
    Add initial  spans to a profile based on a model.
    """
    prediction = model.predict(image)
    full_annotations = full_annotation_array_to_list(prediction.image)
    spans = []
    label_to_color = {}
    for k, v in LABEL_CATEGORIES.items():
        label_to_color[v['pyval']] = v['color']

    for annotation in full_annotations:
        dye, x0, x1, category, _ = annotation

        artist = axs[dye].axvspan(x0, x1,
                                  color=label_to_color[CATEGORIES[category]],
                                  alpha=0.3)

        spans.append({"artist": artist, "ax": axs[dye], "category": CATEGORIES[category],
                      "x0": x0, "x1": x1, "peak_idx": -1})

    return spans


def _add_initial_spans(
        axs: List[Axes], peak_ranges: List[List[Tuple[float, float]]]
) -> List[Dict[str, object]]:
    """
    Add initial unlabeled spans per axis based on peak index ranges.

    Parameters
    ----------
    peak_ranges : list of list of (start_idx, end_idx)
        peak_ranges[i] for a single dye, corresponding to the axis self.ax[i].
    axs: list of axes
    """
    spans = []
    for i, ax in enumerate(axs):
        for start_x, end_x, peak_idx in peak_ranges[i]:
            x0, x1 = sorted([start_x, end_x])
            artist = ax.axvspan(x0, x1, color="gray", alpha=0.2)  # type: ignore[index]
            spans.append({"artist": artist, "ax": ax, "category": None,
                          'x0': x0, 'x1': x1, "peak_idx": peak_idx})

    return spans


def scan_to_bp(x):
    return OFFSET + SLOPE * x


def bp_to_scan(bp):
    return (bp - OFFSET) / SLOPE


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
                        color='green', alpha=.4)

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
