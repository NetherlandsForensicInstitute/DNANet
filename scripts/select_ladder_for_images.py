import argparse
import csv
import os
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks

from DNAnet.data.data_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage, Ladder
from DNAnet.data.utils import basepair_to_pixel
from config_io import load_dataset

OTHER_KITS = ("ppy23", "minifiler", "hdplex")  # other than PPF6C


def get_ladder_paths(im: HIDImage) -> List[Optional[str]]:
    """
    Find all paths to ladder files in the same folder as the provided image.
    """
    return [im.path.parent / file for file in os.listdir(im.path.parent) if
            ("ladder" in file.lower()
             and not any([kit in file.lower() for kit in OTHER_KITS]))]


def get_best_ladder_path(image: HIDImage, default_panel: Panel) -> Optional[str]:
    """
    Return the path of the best fitting ladder for the HIDImage.

    For every called allele in the image's analyst annotation, we retrieve the rfu (peak height)
    data of that allele bin. We check from the rfu values whether there exists a peak in the allele
    bin, and count how often a peak is found. This is done for every ladder. The (pixel) location
    of the allele bin may differ per ladder, therefore there might not always be a peak present in
    the bin. We return the ladder for which most peaks are found.
    If there are no ladder paths available for the image or no paths result in a valid ladder,
    return None.

    TODO: implement an algorithm to find a ladder when an HIDImage is not annotated.
    """
    best_ladder_path = None
    max_peaks_found = 0
    # retrieve paths for all possible ladders
    for path in get_ladder_paths(image):
        ladder = Ladder(path, default_panel)
        if not ladder._panel:
            # proceed if the ladder has no valid adjusted panel
            continue

        n_peaks_found = 0
        for marker in image.meta['called_alleles']:
            for allele in marker.alleles:
                # for every annotated peak, find the allele bin in terms of base pairs
                _, bp_left, bp_right = ladder._panel.get_allele_info(marker.name, allele.name)
                # translate base pairs to pixels using the image's scaler
                pix_left, pix_right = basepair_to_pixel(image.scaler, bp_left), \
                                      basepair_to_pixel(image.scaler, bp_right)
                # retrieve the rfu data for the bin by inspecting the image's data
                allele_rfu_data = image.data[marker.dye_row, int(pix_left):int(pix_right), 0]
                # check if there is a peak located inside the bin
                peak, _ = find_peaks(allele_rfu_data, height=0.8*np.max(allele_rfu_data))
                if peak.size > 0:
                    n_peaks_found += 1
        if n_peaks_found > max_peaks_found:
            # We want to return the ladder with most peaks found, so save the ladder file
            # path if we have found a new maximum
            max_peaks_found = n_peaks_found
            best_ladder_path = ladder.path
    return best_ladder_path


def run(data_config: str, output_path: str):
    """
    Write a csv file with the best ladder path for every image in the dataset. Note that this
    is only done for images that have annotations, a non-empty .data attribute and called alleles
    in the .meta attribute (i.e. that survived all filtering applied when loading a Dataset).
    """
    dataset = load_dataset(data_config)
    panel = Panel('resources/data/SGPanel_PPF6C_SPOOR.xml')
    paths = [[image.path.stem, get_best_ladder_path(image, panel)] for image in dataset]
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'ladder_path'])
        writer.writerows(paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data-config',
        type=str,
        required=True,
        help="The path to a .yaml file (or the basename without an extension "
             "if located in `./config/data/`) containing the configuration "
             "for the dataset to evaluate the model on"
    )
    parser.add_argument(
        '-o',
        '--output-path',
        type=str,
        required=False,
        help="The output path to the the csv file to write the result to.",
        default="resources/data/2p_5p_Dataset_NFI/best_ladder_paths.csv"
    )
    args = parser.parse_args()
    run(args.data_config, args.output_path)
