"""
outputs some simple statistics about your annotations

-how many annotations
-per type of label
-per type of profile

-for data with ground truth, what was the annotator 'performance'
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from DNAnet.allele_callers import NearestBasePairCaller, scan_point_annotation
from DNAnet.data.data_models import Marker
from DNAnet.data.data_models.hid_dataset import HIDDataset
from DNAnet.evaluation.utils import flatten_marker_list_to_locusallelename_list
from config_io import load_dataset

# Get the parent directory of the script folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DNAnet.constants import LABELTOOL_VERSION
from DNAnet.data.utils import full_annotation_array_to_list
from DNAnet.evaluation.visualizations import DNA_CHANNELS

CATEGORIES = ["", "Allele", "Stutter", "PullUp", "BleedThrough", "Spike", "DyeBlob", "Artefact",
              "Unclear", "Shoulder", "ForeignDNA", "OverloadingArtifact"]


def parse_csv_file(csv_file: str):
    parsed_lines = []
    blacklist = set()
    with open(csv_file, 'r') as file:
        try:
            csv_file = csv.reader(file, delimiter=',')

            for num, line in enumerate(csv_file):
                if num == 0:
                    header = line
                else:
                    annotator, sample_name, dye, x0, x1, rfu, category = line[0], \
                        line[2], line[3], line[4], line[5], line[6], line[7]
                    if rfu == '':
                        rfu = -1
                    dye_idx = DNA_CHANNELS.index(dye)
                    if dye_idx == 5:
                        blacklist.add(sample_name)
                    elif category:  # exclude empty strings
                        parsed_lines.append((annotator,
                                             sample_name,
                                             dye, int(x0), int(x1), int(rfu), category))
        except PermissionError:
            return []
    # print('ILS annotations for: ', len(blacklist), blacklist)
    return parsed_lines


def parse_full_annotations(annotation_folder_path: str,
                           per_annotator: bool = False,
                           ) -> dict[str, np.array]:
    """
    Reads all the individual annotations and maps them to a dict (of sample name, or
    (annotator, sample_name) to array of size is [num dyes, scan points]

    NB if two peaks partly overlap, the resulting array will (incorrectly) merge this to one peak.
    """
    lines_dict = read_and_group_annotations(annotation_folder_path, per_annotator)

    sample_arrays = defaultdict(partial(np.zeros, (5, 4096)))
    for key, lines in lines_dict.items():
        for ann in lines:
            sample_arrays[key][ann.dye_index, int(ann.start):int(ann.end)] = CATEGORIES.index(
                ann.label)
    return sample_arrays


def parse_full_annotations_to_called_alleles(
        annotation_folder_path: str,
        images: HIDDataset,
        per_annotator: bool = False,
) -> dict[str, dict[str, list[Marker]]]:
    """
    Reads all the individual annotations and maps them to a dict (of sample name, or
    (annotator, sample_name) to a dict of list of called alleles per label type.

    NB all artifacts are mapped to the best fitting Allele
    """
    lines_dict = read_and_group_annotations(annotation_folder_path, per_annotator)

    called_alleles_per_sample = {}
    for key, lines in lines_dict.items():
        if '1A4' in key:
            pass
        if '4B5_D04_10' in key or '4B5' in key:
            print(lines)
        # make a list per category
        categories = set(l.label for l in lines)
        nbc = NearestBasePairCaller()
        alleles_per_label = {}
        for image in images:
            if is_image_from_key(image, key, per_annotator):
                for cat in categories:
                    alleles_per_label[cat] = nbc.call_alleles_from_scan_points_annotations(
                        image,
                        [line for line in lines if line.label == cat],
                        aggregate=False,
                        warn_for_multiple_peaks=cat=='Allele' or cat=='Stutter'
                    )

                called_alleles_per_sample[key] = alleles_per_label
                continue

    return called_alleles_per_sample


def is_image_from_key(image, key, per_annotator):
    return image.path.stem.split('/')[-1] == key or (per_annotator
                                                     and image.path.stem.split('/')[-1] == key[1])


def read_and_group_annotations(annotation_folder_path, per_annotator):
    """
    gives back all annotations per sample, and optionally per annotator
    NB ignores all annotations of 1 pixel wide, these are assumed to be erroneous
    """
    lines_dict = defaultdict(list)
    csv_files = Path(annotation_folder_path).rglob('*.csv')
    for csv_file in csv_files:
        # gather the annotations per sample or per sample and annotator
        parsed_lines = parse_csv_file(csv_file)
        for parsed_line in parsed_lines:
            annotator, sample_name, dye, x0, x1, rfu, category = parsed_line
            if x0 == x1:
                # ignore annotations that are too small - these are not made by humans
                # but pre-annotations that could not be adjusted.
                continue
            if per_annotator:
                lines_dict[(annotator, sample_name)].append(
                    scan_point_annotation(DNA_CHANNELS.index(dye), int(x0), int(x1), category))
            else:
                lines_dict[sample_name].append(
                    scan_point_annotation(DNA_CHANNELS.index(dye), int(x0), int(x1), category))
    return lines_dict


def array_to_csv(sample_name: str,
                 full_annotations_array: np.ndarray,
                 per_annotator=False,
                 rfu_array: np.ndarray = None) -> list[list]:
    """
    takes an array of predictions such as made by a model, and converts
    it to the rows such as needed for the csv labeltool
    """
    annotations = full_annotation_array_to_list(full_annotations_array, rfu_array)
    if per_annotator:
        # sample name is a tuple of annotator and sample name.
        annotator, sample_name = sample_name
    else:
        annotator = 'annotation_statistics_script'
    return [[annotator,
             '',
             sample_name,
             DNA_CHANNELS[row_idx],
             start,
             end,
             max_rfu,
             CATEGORIES[int(label)],
             LABELTOOL_VERSION] for (row_idx, start, end, label, max_rfu) in annotations]


def write_predictions_to_csv(sample_arrays: dict[str, np.ndarray],
                             output_file: str,
                             per_annotator=False,
                             hidimages: HIDDataset = None):
    header = ['user', 'date', 'profile', 'dye', 'x0', 'x1', 'max_rfu', 'category', 'version']
    result = []
    for sample, array in sample_arrays.items():
        if hidimages:
            for hidimage in hidimages:
                if is_image_from_key(hidimage, sample, per_annotator):
                    result.extend(array_to_csv(sample, array,
                                               per_annotator=per_annotator,
                                               rfu_array=hidimage.data))
        else:
            result.extend(array_to_csv(sample, array, per_annotator=per_annotator))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(result)


def classify_sample_based_on_name(sample_name: str):
    """
    Lab-specific function to specify a type of sample from the given sample name.
    """
    if sample_name.split("_")[2][0] in ("A", "Z"):
        return 'trace'
    elif sample_name.split("_")[2][0] in ("W", "R", "N"):  # N is upgrades
        return 'ref'
    elif sample_name.split("_")[2][0].isdigit():
        # this corresponds to the data provided in resources/data/
        return 'rd'
    elif sample_name.split("_")[1] == 'rerun':  # eg 2E3_rerun_B01
        return 'rd'
    else:
        raise ValueError(f'cannot parse {sample_name}')


def publish_statistics_of_annotations(annotation_file_path: str, plots=True):
    """
    Compute and print some basic statistics about the annotations in a csv file
    """
    parsed_lines = parse_csv_file(annotation_file_path)
    annotation_widths = []
    sample_values = defaultdict(lambda: [0] * (len(CATEGORIES)))
    for parsed_line in parsed_lines:
        _, sample_name, dye, x0, x1, rfu, category = parsed_line
        cat = classify_sample_based_on_name(sample_name)
        sample_values[sample_name][CATEGORIES.index(category)] += 1
        sample_values[sample_name][0] = cat
        annotation_widths.append([category, int(x1) - int(x0), cat])
    df_samples = pd.DataFrame(sample_values).T
    print(df_samples)
    df_samples.columns = ['sample_type'] + CATEGORIES[1:]

    df_annotations = pd.DataFrame(annotation_widths)
    df_annotations.columns = ['category', 'width', 'sample_type']

    print(f'total number of annotations: {len(parsed_lines)}')
    for category in CATEGORIES[1:]:
        subset = [p for p in parsed_lines if p[5] == category]
        print(f'    {category}: {len(subset)}')

    print(f'total number of sample: {df_samples.sample_type.value_counts().sum()}')
    print(df_samples.sample_type.value_counts())

    print(f'Average number of annotations per profile type: ')
    print(df_samples.groupby(['sample_type']).mean())

    if not plots:
        return
    # Get all columns except the last one for histograms
    color_col = 'sample_type'

    # Create plots for the number of annotations per category
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()

    for idx, col in enumerate(df_samples.columns):
        if col != color_col:
            sns.histplot(data=df_samples, x=col, hue=color_col, bins=20, kde=True, ax=axes[idx])
        else:
            sns.histplot(data=df_samples, x=col, bins=20, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel('Count')
        axes[idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('output/annotations per category')

    # Create plots for the widths per category
    color_col = 'sample_type'
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()
    for idx, col in enumerate(CATEGORIES[1:]):
        sns.histplot(data=df_annotations[df_annotations['category'] == col],
                     x='width', hue=color_col, bins=range(0, 80, 5), kde=True, ax=axes[idx])
        axes[idx].set_title(f'Widths for {col}')
        axes[idx].set_xlabel('Count')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlim([0, 80])
    plt.tight_layout()
    plt.savefig('output/widths per category')


def publish_performance_on_ground_truth(annotation_folder_path: str):
    """
    filters profiles for those with ground truth, publishes the confusion matrix
    """

    profile_data = load_dataset("dnanet_rd_annotation.yaml")
    called_alleles = parse_full_annotations_to_called_alleles(annotation_folder_path,
                                                              profile_data,
                                                              per_annotator=False)
    results = {}
    n_profile = 0
    for profile in profile_data:
        sample_name = profile.path.name.strip('.hid')
        if sample_name in called_alleles:
            if '1A4' in sample_name:
                pass # one annotation for a group??
            true_positive = defaultdict(int)
            false_positive = defaultdict(int)
            false_negative = defaultdict(int)
            true_negative = defaultdict(int)

            true_alleles = flatten_marker_list_to_locusallelename_list(
                profile.meta['called_alleles'],
            )

            # in this order, see what true alleles we cumulatively recall
            labels = ['Allele', 'Stutter', 'BleedThrough', 'Artefact', 'Shoulder', 'Spike']
            predicted_alleles = set()
            for idx, label in enumerate(labels):
                if label in called_alleles[sample_name]:
                    predicted_alleles.update(flatten_marker_list_to_locusallelename_list(
                        called_alleles[sample_name][label],
                        # remove duplicates - these are due to copying of annotation files,
                        # not actual dual
                        # annotations.
                        remove_duplicates=True,
                    ))
                    # remove these, they do not exist in true alleles
                    predicted_alleles = {p for p in predicted_alleles if
                                         not (p.startswith('AMEL') or p.startswith('DYS'))}

                    false_negative[label], false_positive[label], true_negative[label], \
                    true_positive[label] = compute_metrics(
                        predicted_alleles, true_alleles)
                    if label == 'Allele':
                        # print all errors
                        missed = true_alleles.difference(predicted_alleles)
                        spurious = predicted_alleles.difference(true_alleles)


                        if missed or spurious:
                            print(sample_name)
                            print('missed alleles: ', sorted(missed))
                            print('incorrect allele calls: ', sorted(spurious))

                            # extra attention for possible wrong allele calling
                            add_ones = [s for s in spurious if s.endswith('.1') or s.endswith('.3')]
                            for a in spurious:
                                if a.endswith('.1'):
                                    if a[:-2] in missed:
                                        print('----------------', a, missed, spurious)
                                if a.endswith('.3'):
                                    no = int(a[-4:-2].strip('_'))
                                    if a[:-4]+str(no+1) in missed:
                                        print('----------------', a, missed, spurious)

                else:
                    # set the numbers to be equal to the previous label (assuming the first label
                    # 'Allele' is always present
                    false_negative[label] = false_negative[labels[idx - 1]]
                    false_positive[label] = false_positive[labels[idx - 1]]
                    true_negative[label] = true_negative[labels[idx - 1]]
                    true_positive[label] = true_positive[labels[idx - 1]]

            # also compare to LT annotations
            original_called_alleles = flatten_marker_list_to_locusallelename_list(
                profile.meta['called_alleles_manual'])
            original_called_alleles = {p for p in original_called_alleles if
                                       not (p.startswith('AMEL') or p.startswith('DYS'))}

            false_negative['SOP'], false_positive['SOP'], true_negative['SOP'], true_positive[
                'SOP'] = compute_metrics(
                original_called_alleles, true_alleles)

            results[sample_name] = \
                [recall_precision(false_negative, true_positive, label)  # recall
                 for label in labels] + [
                    recall_precision(false_negative, true_positive, 'SOP'),  # recall
                    recall_precision(false_positive, true_positive, 'Allele'),  # precision
                    recall_precision(false_positive, true_positive, 'SOP'),  # precision
                ]

            n_profile += 1

    df = pd.DataFrame(results).T
    df.columns = ([f'recall {label}' for label in labels] +
                  ['recall SOP', 'precision Allele', 'precision SOP'])
    with pd.option_context('display.width', None,
                           'display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(df)
    print(f'n profiles: {n_profile}')
    print(df.mean())


def recall_precision(false_negative, true_positive, key):
    if true_positive[key] == 0:
        return 0
    return true_positive[key] / (true_positive[key] + false_negative[key])


def compute_metrics(predicted_alleles, true_alleles):
    true_positive = len(true_alleles.intersection(predicted_alleles))
    false_positive = len(predicted_alleles) - true_positive
    false_negative = len(true_alleles) - true_positive
    true_negative = len(true_alleles.union(predicted_alleles)) - len(true_alleles) - false_positive
    return false_negative, false_positive, true_negative, true_positive

def run(annotation_folder_path: str, output_file: str):

    # first, bring all separate annotation files together into one file (to remove duplicates)
    sample_arrays = parse_full_annotations(annotation_folder_path)
    publish_statistics_of_annotations(output_file, plots=False)
    write_predictions_to_csv(sample_arrays, output_file)

    # for data with known ground truth, output performance of annotators
    publish_performance_on_ground_truth(annotation_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a',
        '--annotation-folder-path',
        type=str,
        required=True,
        help='path to the folder containing the annotation csv files'
    )
    parser.add_argument(
        '-o',
        '--output-file',
        type=str,
        required=True,
        default='output/annotations_all.csv',
        help='path to the output csv file where the combined annotations will be written to'
    )
    args = parser.parse_args()
    run(args.annotation_folder_path, args.output_file)

