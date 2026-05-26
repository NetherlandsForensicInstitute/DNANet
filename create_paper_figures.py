import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from DNAnet.models.base import Model
from config_io import load_dataset, load_model
from DNAnet.data.data_models.dna_models import Marker, Panel
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.evaluation.visualizations import DNA_CHANNELS, _get_marker_bin
from DNAnet.models.prediction import Prediction

# Set up matplotlib parameters for the paper figures text size
plt.rcParams.update(
    {
        "axes.titlesize": 24,  # Title size
        "axes.labelsize": 24,  # X and Y axis label size
        "xtick.labelsize": 24,  # X tick label size
        "ytick.labelsize": 24,  # Y tick label size
        "legend.fontsize": 20,  # Legend font size
        "figure.titlesize": 24,  # Figure suptitle size
    }
)

# Define colors for different annotations
ANNOTATION_COLORS = {
    "Ground Truth": "blue",
    "Analyst Annotation": "green",
    "Model Prediction": "orange",
}
# Interpolation function to convert scan points to base pairs
SCAN_TO_BASE = scipy.interpolate.interp1d(
    [0, 4096], [65, 475], fill_value="extrapolate"
)

# Set up logging
logger = logging.getLogger("Figure Generator")
logger.setLevel(logging.INFO)


def add_bin_info(called_alleles: List[Marker], panel: Panel) -> List[Marker]:
    """Using a Panel object, add bin information to the called alleles.

    Args:
        called_alleles (List[Marker]): The list of Markers with missing allele bin information.
        panel (Panel): The (scaler) Panel object containing the allele information.

    Returns:
        List[Marker]: The updated list of Markers with base pair and bin information added.
    """
    for marker in called_alleles:
        for allele in marker.alleles:
            bin_info = panel.get_allele_basepair_and_bins(
                marker_name=marker.name, allele_name=allele.name
            )
            allele.base_pair, allele.left_bin, allele.right_bin = bin_info
    return called_alleles


def get_alleles(
    image: HIDImage, prediction: Prediction, only_autosomal: bool = True
) -> Dict[str, Tuple[List[Marker], np.ndarray]]:
    """Get the alleles from the image and prediction, and prepare them for plotting.
    This function retrieves the called alleles from the prediction and the image metadata, for the analyst annotations and ground truth.
    It also prepares the binary segmentation maps for the alleles, which are used for plotting.

    Args:
        image (HIDImage): The HIDImage object containing the image data and metadata.
        prediction (Prediction): The Prediction object containing the model's predictions.
        only_autosomal (bool, optional): Whether to exclude non-autosomal markers (e.g. AMEL). Defaults to True.

    Returns:
        Dict[str, Tuple[List[Marker], np.ndarray]]: A dictionary containing the called alleles and their corresponding binary segmentation maps.
    """
    # Get alleles and add bin info
    prediction_alleles = prediction.called_alleles if prediction.called_alleles else []
    if manual_alleles := image.meta.get("called_alleles_manual", None):
        analyst_alleles = manual_alleles
        ground_truth_alleles = image.meta["called_alleles"]
    else:
        analyst_alleles = image.meta["called_alleles"]
        ground_truth_alleles = []

    prediction_alleles: List[Marker] = add_bin_info(prediction_alleles, image._panel)
    analyst_alleles: List[Marker] = add_bin_info(analyst_alleles, image._panel)
    if only_autosomal:
        analyst_alleles = [marker for marker in analyst_alleles if marker.is_autosomal]

    ground_truth_alleles = (
        []
        if not ground_truth_alleles
        else add_bin_info(ground_truth_alleles, image._panel)
    )

    # Get a binary map of the called alleles
    prediction_allele_segmentation = HIDImage._get_segmentation(
        image.scaler, prediction_alleles, (5, 4096, 1)
    )
    analyst_allele_segmentation = HIDImage._get_segmentation(
        image.scaler, analyst_alleles, (5, 4096, 1)
    )
    ground_truth_allele_segmentation = HIDImage._get_segmentation(
        image.scaler, ground_truth_alleles, (5, 4096, 1)
    )

    allele_dict = {
        "Model Prediction": (prediction_alleles, prediction_allele_segmentation),
        "Analyst Annotation": (analyst_alleles, analyst_allele_segmentation),
    }
    if ground_truth_alleles:
        allele_dict["Ground Truth"] = (
            ground_truth_alleles,
            ground_truth_allele_segmentation,
        )
    return allele_dict


def plot_allele_profile(
    image: HIDImage,
    prediction: Prediction,
    show_probability: bool = True,
    show_annotations: bool = True,
    marker_selection: Tuple[str, Tuple[int, np.ndarray]] = None,
    add_suptitle: bool = False,
) -> plt.Figure:
    """Plot the allele profile of a HIDImage with the model's predictions and annotations.

    Args:
        image (HIDImage): The HIDImage object to plot.
        prediction (Prediction):
            The Prediction object containing the model's predictions on the image.
        show_probability (bool, optional):
            Whether to plot the model's prediction as a probability-line.
            Defaults to True.
        show_annotations (bool, optional):
            Whether to show all called allele annotations for the three possible annotators.
            Defaults to True.
        marker_selection (Tuple[str, Tuple[int, np.ndarray]], optional):
            A specific marker to zoom-in on for the plot, takes the whole profile if left blank.
            Expects a tuple with the marker name and a tuple of the dye row and the bin range.
            Defaults to None.
        add_suptitle (bool, optional):
            Whether to add the image's name to the plot.
            Defaults to False.

    Returns:
        plt.Figure: _description_
    """
    allele_dict: Dict[str, Tuple[List[Marker], np.ndarray]] = get_alleles(
        image, prediction
    )
    if marker_selection:
        marker_name, (dye_row, marker_bin) = marker_selection
        fig, axes = plt.subplots(nrows=1, figsize=(14, 8), dpi=400)
        axes = [axes]
    else:
        fig, axes = plt.subplots(nrows=5, figsize=(20, 16), dpi=400)
    if add_suptitle:
        fig.suptitle(f"HID: {image.path.stem}")

    x_values = np.arange(image.data.shape[1])
    x_values = SCAN_TO_BASE(x_values)
    if marker_selection:
        x_values = x_values[marker_bin]

    for idx, ax in enumerate(
        axes,
    ):
        if marker_selection:
            ax.set_title(marker_name)
        ax.set_xlabel("Base Pair")
        # Plot the RFU values
        ax.plot(
            x_values,
            image.data[idx, :, 0]
            if not marker_selection
            else image.data[dye_row, marker_bin, 0],
            color=DNA_CHANNELS[idx] if not marker_selection else DNA_CHANNELS[dye_row],
        )
        ax.set_ylabel("RFU")

        if show_annotations:
            # Add extra space for the annotations
            ymin, ymax = ax.get_ylim()
            extra_space = (ymax - ymin) * 0.3
            ax.set_ylim(ymin - extra_space, ymax)
            ax.set_yticks([tick for tick in ax.get_yticks() if 0 <= tick])

        if show_probability:
            # Plot the model's probabilities
            rax = ax.twinx()
            rax.plot(
                x_values,
                prediction.image[idx, :, 0]
                if not marker_selection
                else prediction.image[dye_row, marker_bin, 0],
                color="orange",
                alpha=0.6,
            )
            # Add a horizontal line at the prediction threshold
            rax.hlines(
                y=0.5,
                xmin=rax.get_xlim()[0],
                xmax=rax.get_xlim()[1],
                color="grey",
                linestyle="dashed",
                alpha=0.3,
                label="Model prediction Threshold",
            )
            rax.set_ylabel("Probability")
            rax.set_yticks([tick for tick in rax.get_yticks() if 0 <= tick <= 1.01])

            # Set the RFU plot above the probability plot
            ax.set_zorder(rax.get_zorder() + 1)
            ax.set_frame_on(False)

            if show_annotations:
                # Also add annotation space for this axes
                ymin, ymax = rax.get_ylim()
                extra_space = (ymax - ymin) * 0.3
                rax.set_ylim(ymin - extra_space, ymax)

        if show_annotations:
            # Create an overlay for the annotations
            overlay = ax.figure.add_axes(ax.get_position(), frameon=False)
            overlay.set_xlim(ax.get_xlim())
            overlay.axis("off")

            # For each (available) annotation, plot vlines on the called alleles spot
            for annotation_index, (
                annotation_name,
                (_, allele_segmentation),
            ) in enumerate(allele_dict.items()):
                line_dist = 50
                block_height = 200
                y_max, y_min = (
                    (annotation_index) * -block_height - line_dist,
                    (annotation_index + 1) * -block_height,
                )
                overlay.set_ylim(top=4000, bottom=-1000)
                overlay.vlines(
                    x=SCAN_TO_BASE(
                        np.nonzero(
                            allele_segmentation[idx, :, 0]
                            if not marker_selection
                            else allele_segmentation[dye_row, :, 0]
                        )
                    ),
                    ymin=y_min,
                    ymax=y_max,
                    color=ANNOTATION_COLORS[annotation_name],
                    label=annotation_name,
                    linewidth=3,
                )

            custom_lines = [
                Line2D(
                    [0],
                    [0],
                    color=ANNOTATION_COLORS[annotation_name],
                    label=annotation_name,
                )
                for annotation_name in allele_dict.keys()
            ]

            # Place legend slightly above the topmost axis
            axes[0].legend(
                handles=custom_lines,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.05),
                ncol=3,
                frameon=False,
            )
    return fig


def get_marker_ranges(
    image: HIDImage, tail_size: int = 5
) -> Dict[str, Tuple[int, np.ndarray]]:
    """Using the image's panel, get the ranges of the markers in the image.
    This function retrieves the ranges of the markers in the image,
    which are used for zooming in on specific markers.

    Args:
        image (HIDImage): The HIDImage object containing the image data and metadata.
        tail_size (int, optional):
            Extra space that's added on the left and right side of a marker's bin.
            Defaults to 5.

    Returns:
        Dict[str, Tuple[int, np.ndarray]]: A dictionary containing the marker names as keys,
        and a tuple of the dye row and the bin range as values.
    """
    marker_ranges = {}
    for marker in image._panel._panel:
        marker_name = marker.name
        marker_bin = _get_marker_bin(marker)
        scanpoint_bin = tuple(np.argmin(np.abs(image._scaler - marker_bin), axis=1))
        scanpoint_bin = (
            max(0, scanpoint_bin[0] - tail_size),
            min(4096, scanpoint_bin[1] + tail_size),
        )
        marker_ranges[marker_name] = (marker.dye_row, np.arange(*scanpoint_bin))
    return marker_ranges


def save_figure(fig: plt.Figure, path: str):
    """Save the figure to the specified path.

    Args:
        fig (plt.Figure): The figure to save.
        path (str): The path to save the figure to.
    """
    fig.savefig(path)
    plt.close(fig)


def generate_figures(
    model: Model,
    dataset: HIDDataset,
    selected_hids_markers: List[Tuple[str, Optional[str]]],
    output_dir="./figures",
):
    """Generate figures for the selected HID images and markers.

    Args:
        model (Model): The trained model to use for predictions.
        dataset (HIDDataset): The dataset containing the HID images.
        selected_hids_markers (List[Tuple[str, Optional[str]]]):
            The selected HID images and their corresponding markers.
            Should be a list of tuples with the HID image name and (optionally) the marker name.
        output_dir (str, optional): The directory to save the figures to. Defaults to "./figures".
    """
    Path(output_dir).mkdir(exist_ok=True)

    figure_images = [
        (hid_name, marker, dataset.get_hid_image_by_name(hid_name))
        for hid_name, marker in selected_hids_markers
    ]
    for hid_name, marker, image in figure_images:
        image_id = image.path.stem
        if image_id != hid_name:
            continue

        prediction = model.predict(image)
        marker_ranges = get_marker_ranges(image)

        # Full profile figure
        full_profile_fig = plot_allele_profile(
            image, prediction, show_probability=False, show_annotations=True
        )
        save_figure(full_profile_fig, f"{output_dir}/{image_id}-full_profile.png")

        # Marker-specific figure
        if marker in marker_ranges:
            marker_figure = plot_allele_profile(
                image,
                prediction,
                marker_selection=(marker, marker_ranges[marker]),
            )
            save_figure(marker_figure, f"{output_dir}/{image_id}-{marker}.png")


def create_bin_type_plot(dataset: HIDDataset) -> None:
    """Create a plot showing the different types of annotations used in the paper.

    Args:
        dataset (HIDDataset): The dataset containing the HID images.
    """
    # Create three columns for each annotation type
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4), dpi=400)

    # We used a specific image for this plot, so we load it directly
    _data = dataset.get_hid_image_by_name("1A2_E01_13")._data
    letters = ["A", "B", "C"]
    # The highlight ranges explaining the different bin types
    highlight_ranges = [(15, 22), (19, 19), (8, 32)]

    # For each axis, plot the data and highlight the ranges
    for ax, hrange, letter in zip(ax, highlight_ranges, letters):
        ax.plot(np.arange(40), _data[2, 290:330, 0], color="black")
        ax.axvspan(*hrange, color="green", alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="left",
            color="red",
        )
    fig.savefig("figures/bin_type_plot.png", bbox_inches="tight")


if __name__ == "__main__":
    model = load_model("resources/model/current_best_unet/")
    dataset = load_dataset("config/data/dnanet_rd.yaml")

    paper_figures = [
        ("1E3_rerun_F04_16", "SE33"),
        ("1E3_rerun_F04_16", "D10S1248"),
        ("3D2_A07_01", "D8S1179"),
    ]

    generate_figures(model, dataset, paper_figures)
    create_bin_type_plot(dataset)
