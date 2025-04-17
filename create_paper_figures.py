import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from matplotlib.lines import Line2D
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy
import scipy.interpolate
from tqdm import tqdm
import yaml

from DNAnet.data.data_models import Marker
from DNAnet.data.data_models.hid_image import HIDImage
from DNAnet.evaluation.visualizations import _get_marker_bin
from DNAnet.models.prediction import Prediction
from config_io import load_dataset, load_model
from copy import deepcopy

logger = logging.getLogger("Figure Generator")
logger.setLevel(logging.INFO)

DNA_CHANNELS = ('blue', 'green', 'black', 'red', 'purple', 'orange')
ANNOTATION_COLORS = {
    "Ground Truth": "blue",
    "Analyst Annotation": "green",
    "Model Prediction": "orange"
}
SCAN_TO_BASE = scipy.interpolate.interp1d([0, 4096],[65, 475], fill_value="extrapolate")

def add_bin_info(called_alleles, panel):
    for marker in called_alleles:
        for allele in marker.alleles:
            bin_info = panel.get_allele_info(
                marker_name=marker.name,
                allele_name=allele.name
            )
            allele.base_pair, allele.left_bin, allele.right_bin = bin_info
    return called_alleles


def get_alleles(image, prediction):
    # Get alleles and add bin info
    prediction_alleles = prediction.meta["called_alleles"]
    if (manual_alleles := image.meta.get("called_alleles_manual", None)):
        analyst_alleles = manual_alleles
        ground_truth_alleles = image.meta["called_alleles"]
    else:
        analyst_alleles = image.meta["called_alleles"]
        ground_truth_alleles = []
    
    prediction_alleles = add_bin_info(prediction_alleles, image._panel)
    analyst_alleles = add_bin_info(analyst_alleles, image._panel)
    ground_truth_alleles = [] if not ground_truth_alleles else add_bin_info(ground_truth_alleles, image._panel)
    
    # Get a binary map of the called alleles
    prediction_allele_segmentation = HIDImage._get_segmentation(image.scaler, prediction_alleles, (5, 4096, 1))
    analyst_allele_segmentation = HIDImage._get_segmentation(image.scaler, analyst_alleles, (5, 4096, 1))
    ground_truth_allele_segmentation = HIDImage._get_segmentation(image.scaler, ground_truth_alleles, (5, 4096, 1))
    
    allele_dict = {
        "Model Prediction": (prediction_alleles, prediction_allele_segmentation),
        "Analyst Annotation": (analyst_alleles, analyst_allele_segmentation)
    }
    if ground_truth_alleles:
        allele_dict["Ground Truth"] = (ground_truth_alleles, ground_truth_allele_segmentation)
    return allele_dict

def create_paper_figure(
    image,
    prediction,
    show_probability: bool = True,
    show_annotations: bool = True,
    marker_selection = None,
    add_title: str = False,
):
    allele_dict = get_alleles(image, prediction)
    if marker_selection:
        marker_name, (dye_row, marker_bin) = marker_selection
        fig, axes = plt.subplots(nrows=1, figsize=(12, 8))
        axes = [axes]
    else:
        fig, axes = plt.subplots(nrows=5, figsize=(20, 20))
    if add_title:
        fig.suptitle(f"HID: {image.path.stem}")
    
    x_values = np.arange(image.data.shape[1])
    x_values = SCAN_TO_BASE(x_values)
    if marker_selection:
        x_values = x_values[marker_bin]
    
    for idx, ax in enumerate(axes,):
        if marker_selection:
            ax.set_title(marker_selection[0])
        ax.set_xlabel("Base Pair")
        # Plot the RFU values
        ax.plot(
            x_values,
            image.data[idx, :, 0] if not marker_selection else image.data[dye_row, marker_bin, 0],
            color=DNA_CHANNELS[idx] if not marker_selection else DNA_CHANNELS[dye_row]
        )
        ax.set_ylabel("RFU")
        
        if show_annotations:
            # Add extra space for the annotations
            ymin, ymax = ax.get_ylim()
            extra_space = (ymax - ymin) * 0.3
            ax.set_ylim(ymin - extra_space, ymax)
        
        if show_probability:
            # Plot the model's probabilities
            rax = ax.twinx()
            rax.plot(
                x_values,
                prediction.image[idx, :, 0] if not marker_selection else prediction.image[dye_row, marker_bin, 0],
                color="orange", 
                alpha=.6
            )
            rax.hlines(
                y=.5, xmin=rax.get_xlim()[0], xmax=rax.get_xlim()[1],
                color="grey", linestyle="dashed", alpha=.3,
                label="Model prediction Threshold"
            )
            rax.set_ylabel("Probability")
        
            # Set the RFU plot above the probability plot
            ax.set_zorder(rax.get_zorder()+1)
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
            # overlay.set_xlim(0, 4096)
            overlay.axis("off")
            
            # For each (available) annotation, plot vlines on the called alleles spot
            for annotation_index, (annotation_name, (_, allele_segmentation)) in enumerate(allele_dict.items()):
                line_dist = 50
                block_height = 200
                y_max, y_min = (
                    (annotation_index)*-block_height - line_dist,
                    (annotation_index+1)*-block_height,
                )
                overlay.set_ylim(
                    top=4000,
                    bottom=-1000
                )
                overlay.vlines(
                    x=SCAN_TO_BASE(np.nonzero(
                        allele_segmentation[idx, :, 0]
                        if not marker_selection
                        else allele_segmentation[dye_row, :, 0]
                    )),
                    ymin=y_min,
                    ymax=y_max,
                    color=ANNOTATION_COLORS[annotation_name],
                    label=annotation_name,
                    linewidth=3,
                )
            
            custom_lines = [
                Line2D([0], [0], color=ANNOTATION_COLORS[annotation_name], label=annotation_name)
                for annotation_name in allele_dict.keys()
            ]

            # Place legend slightly above the topmost axis
            axes[0].legend(
                handles=custom_lines,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.05),
                ncol=3,
                frameon=False
            )
    return fig

if __name__ == "__main__":
    # Create a path to save the figures to
    Path("./figures").mkdir(exist_ok=True)
    
    # Load model, data, and predictions
    model = load_model("resources/model/current_best_unet/")
    dataset = load_dataset("config/data/dnanet_rd.yaml")
    
    # The specific HID and Markers used for the plots in the paper
    paper_figures = {
        "1E3_rerun_F04_16": "SE33",
        "3D2_A07_01": "D8S1179",
    }

    # Loop of the images and filter for the HID's chosen.
    for image in dataset:
        if image.path.stem not in paper_figures.keys():
            continue
        prediction = model.predict(image)
        
        # Plot the full profile
        full_profile_fig = create_paper_figure(
            image,
            prediction,
            show_probability=False,
            show_annotations=True
        )
        full_profile_fig.savefig(f"./figures/{image.path.stem}-full_profile.png")
        
        # Create a mapping for the bin locations of each marker
        marker_ranges = {}
        tail_size = 5
        for marker in image._panel._panel:
            marker_name = marker.name
            marker_bin = _get_marker_bin(marker)
            scanpoint_bin = tuple(np.argmin(np.abs(image._scaler - marker_bin), axis=1))
            scanpoint_bin = scanpoint_bin[0]-tail_size, scanpoint_bin[1]+tail_size
            marker_ranges[marker_name] = (
                marker.dye_row,
                np.arange(*scanpoint_bin)
            )
        
        # Plot the specific marker shown in the paper
        paper_marker = paper_figures[image.path.stem]
        marker_figure = create_paper_figure(
            image, prediction, marker_selection=(paper_marker, marker_ranges[paper_marker])
        )
        marker_figure.savefig(f"./figures/{image.path.stem}-{paper_marker}.png")
