import argparse
import csv
import os
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks
import math

from DNAnet.data.data_models import Panel
from DNAnet.data.data_models.hid_image import HIDImage, Ladder
from config_io import load_dataset


def bp_to_pix(scaler: np.ndarray, bp: float) -> float:
    return np.argmin(np.abs(scaler - bp), axis=1)[0]


def get_ladder_paths(im: HIDImage) -> List[Optional[str]]:
    return [im.path.parent / file for file in os.listdir(im.path.parent) if
            ("ladder" in file.lower()
             and not any([kit in file.lower() for kit in ("ppy23", "minifiler", "hdplex")]))]


def get_best_ladder_path(image: HIDImage, default_panel: Panel) -> Optional[str]:
    best_ladder_path = None
    min_diff = math.inf
    n_called_alleles = sum([len(m.alleles) for m in image.meta['called_alleles']])
    # retrieve paths for all possible ladders
    for path in get_ladder_paths(image):
        ladder = Ladder(path, default_panel)
        if not ladder._panel:
            # proceed if the ladder has no valid adjusted panel
            continue

        n_alleles_found = 0
        for marker in image.meta['called_alleles']:
            for allele in marker.alleles:
                # for every peak, find the allele bin in base pairs
                _, bp_left, bp_right = ladder._panel.get_allele_info(marker.name, allele.name)
                # translate base pairs to pixels using the image's scaler
                pix_left, pix_right = bp_to_pix(image.scaler, bp_left), \
                                      bp_to_pix(image.scaler, bp_right)
                # retrieve the rfu data for that pixel bin
                allele_rfu_data = image.data[marker.dye_row, int(pix_left):int(pix_right), 0]
                # check if there is a peak within the bin
                peak, _ = find_peaks(allele_rfu_data, height=0.8*np.max(allele_rfu_data))
                if peak.size > 0:
                    n_alleles_found += 1
        if n_called_alleles - n_alleles_found < min_diff:
            # We want to return the ladder with minimal difference in peaks, so save the ladder
            # path if we have found a new minimum
            min_diff = n_called_alleles - n_alleles_found
            best_ladder_path = ladder.path
    return best_ladder_path


def run(data_config: str):
    dataset = load_dataset(data_config)
    exit()
    panel = Panel('resources/data/SGPanel_PPF6C_SPOOR.xml')
    paths = [[image.path, get_best_ladder_path(image, panel)] for image in dataset]
    with open('resources/data/2p_5p_Dataset_NFI/best_ladder_paths.csv', 'w', newline='') as f:
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
    args = parser.parse_args()
    run(args.data_config)
