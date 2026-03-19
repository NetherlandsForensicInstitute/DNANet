import argparse
import itertools
import os
from collections import defaultdict
from typing import Iterable
import sys
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_io import load_dataset
from scan_point_annotation_statistics import (
    parse_csv_file,
    parse_full_annotations,
    parse_full_annotations_to_called_alleles,
    write_predictions_to_csv,
)


"""
This script takes an annotations file where several annotators have annotated the same profiles.
It then looks at how comparable those are.


Sample data structure
annotations = [
    ('Barry', '123', 'blue', '144', '152', 'BleedThrough'),
    ('Alice', '123', 'blue', '144', '152', 'TruePeak'),
    ('Carol', '123', 'blue', '145', '153', 'BleedThrough'),
    # Add your annotations here
]

"""

# Color palette for different label types
label_colors = {
    'Allele': 'rgba(46, 204, 113, 0.4)',  # Green
    'Artefact': 'rgba(231, 76, 60, 0.4)',  # Red
    'BleedThrough': 'rgba(52, 152, 219, 0.4)',  # Blue
    'Stutter': 'rgba(155, 89, 182, 0.4)',  # Purple
    'PullUp': 'rgba(149, 165, 166, 0.4)',  # Gray
    'Shoulder': 'rgba(149, 65, 206, 0.4)',
    'Baseline': 'rgba(52, 73, 94, 0.4)',  # Dark gray
}


def parse_annotations(annotations):
    """Group annotations by profile and dye."""
    grouped = defaultdict(lambda: defaultdict(list))
    for annotator, profile_id, dye, start, end, rfu, label in annotations:
        grouped[profile_id][dye].append(
            {
                'annotator': annotator,
                'start': int(start),
                'end': int(end),
                'rfu': rfu,
                'label': label,
            }
        )
    return grouped


def compute_iou(peak1, peak2):
    """Compute Intersection over Union (IoU) between two peaks."""
    start1, end1 = peak1['start'], peak1['end']
    start2, end2 = peak2['start'], peak2['end']

    overlap = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)

    return overlap / union if union > 0 else 0


def match_peaks_optimal(peaks1, peaks2, threshold=0.2):
    """
    Optimally match peaks using Hungarian algorithm.

    Args:
        peaks1, peaks2: Lists of peak dictionaries
        threshold: Minimum IoU to consider a match (default 0.2 for more lenient matching)

    Returns:
        List of (index1, index2, iou) tuples for matched peaks
    """
    if not peaks1 or not peaks2:
        return []

    # Build cost matrix (negative IoU, since Hungarian minimizes)
    n1, n2 = len(peaks1), len(peaks2)
    cost_matrix = np.zeros((n1, n2))

    for i, p1 in enumerate(peaks1):
        for j, p2 in enumerate(peaks2):
            iou = compute_iou(p1, p2)
            # Negative IoU for minimization; penalize below threshold
            cost_matrix[i, j] = -iou if iou >= threshold else 0

    # Find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract matches above threshold
    matches = []
    for i, j in zip(row_ind, col_ind):
        iou = -cost_matrix[i, j]
        if iou >= threshold:
            matches.append((i, j, iou))

    return matches


def compute_pairwise_agreement(annotations, iou_threshold=0.2, use_optimal_matching=True):
    """
    Compute agreement between all annotator pairs.

    Args:
        annotations: List of annotation tuples
        iou_threshold: Minimum IoU to consider peaks matched (default 0.2)
        use_optimal_matching: Use Hungarian algorithm (True) or greedy (False)
    """
    grouped = parse_annotations(annotations)
    annotators = sorted(set(a[0] for a in annotations))

    location_agreement = defaultdict(lambda: {'matches': 0, 'total': 0, 'iou_scores': []})
    label_agreement = defaultdict(lambda: {'matches': 0, 'total': 0})

    for profile_id, dyes in grouped.items():
        for dye, peaks in dyes.items():
            # Group peaks by annotator
            by_annotator = defaultdict(list)
            for peak in peaks:
                by_annotator[peak['annotator']].append(peak)

            # Compare each pair of annotators
            for ann1, ann2 in itertools.combinations(annotators, 2):
                if ann1 not in by_annotator or ann2 not in by_annotator:
                    continue

                peaks1 = by_annotator[ann1]
                peaks2 = by_annotator[ann2]
                pair = tuple(sorted([ann1, ann2]))

                if use_optimal_matching:
                    matches = match_peaks_optimal(peaks1, peaks2, iou_threshold)

                    # Count matched peaks
                    num_matched = len(matches)
                    location_agreement[pair]['matches'] += num_matched
                    location_agreement[pair]['total'] += max(len(peaks1), len(peaks2))

                    # Store IoU scores
                    for i, j, iou in matches:
                        location_agreement[pair]['iou_scores'].append(iou)
                        # Check label agreement
                        if peaks1[i]['label'] == peaks2[j]['label']:
                            label_agreement[pair]['matches'] += 1
                        label_agreement[pair]['total'] += 1
                else:
                    # Original greedy matching
                    matched = set()
                    for p1 in peaks1:
                        for i, p2 in enumerate(peaks2):
                            if i not in matched and compute_iou(p1, p2) >= iou_threshold:
                                matched.add(i)
                                location_agreement[pair]['matches'] += 1
                                location_agreement[pair]['total'] += 1

                                if p1['label'] == p2['label']:
                                    label_agreement[pair]['matches'] += 1
                                label_agreement[pair]['total'] += 1
                                break
                        else:
                            location_agreement[pair]['total'] += 1

                    location_agreement[pair]['total'] += len(peaks2) - len(matched)

    return location_agreement, label_agreement, annotators


def print_results(location_agreement, label_agreement, annotators):
    """Print agreement statistics."""
    print('INTER-ANNOTATOR AGREEMENT\n')

    print('Location Agreement (% of peaks that overlap):')
    print('-' * 60)
    for pair in sorted(location_agreement.keys()):
        stats = location_agreement[pair]
        pct = 100 * stats['matches'] / stats['total'] if stats['total'] > 0 else 0
        avg_iou = np.mean(stats['iou_scores']) if stats['iou_scores'] else 0
        print(
            f'{pair[0]} vs {pair[1]}: {pct:.1f}% ({stats["matches"]}/{stats["total"]}) | Avg IoU: {avg_iou:.2f}'
        )

    print('\nLabel Agreement (% of matched peaks with same label):')
    print('-' * 60)
    for pair in sorted(label_agreement.keys()):
        stats = label_agreement[pair]
        pct = 100 * stats['matches'] / stats['total'] if stats['total'] > 0 else 0
        print(f'{pair[0]} vs {pair[1]}: {pct:.1f}% ({stats["matches"]}/{stats["total"]})')

    # Overall statistics
    total_loc_matches = sum(s['matches'] for s in location_agreement.values())
    total_loc = sum(s['total'] for s in location_agreement.values())
    total_lab_matches = sum(s['matches'] for s in label_agreement.values())
    total_lab = sum(s['total'] for s in label_agreement.values())

    all_ious = []
    for stats in location_agreement.values():
        all_ious.extend(stats['iou_scores'])
    avg_iou_overall = np.mean(all_ious) if all_ious else 0

    print('\nOVERALL:')
    print(
        f'Location agreement: {100 * total_loc_matches / total_loc:.1f}% | Mean IoU: {avg_iou_overall:.2f}'
    )
    print(f'Label agreement: {100 * total_lab_matches / total_lab:.1f}%')


def plot_agreement_heatmaps(location_agreement, label_agreement, annotators, output_append=''):
    """Create heatmaps showing pairwise agreement between annotators."""
    n = len(annotators)
    loc_matrix = np.zeros((n, n))
    lab_matrix = np.zeros((n, n))

    # Create annotator index mapping
    ann_to_idx = {ann: i for i, ann in enumerate(annotators)}

    # Fill matrices
    for pair, stats in location_agreement.items():
        i, j = ann_to_idx[pair[0]], ann_to_idx[pair[1]]
        pct = 100 * stats['matches'] / stats['total'] if stats['total'] > 0 else 0
        loc_matrix[i, j] = pct
        loc_matrix[j, i] = pct

    for pair, stats in label_agreement.items():
        i, j = ann_to_idx[pair[0]], ann_to_idx[pair[1]]
        pct = 100 * stats['matches'] / stats['total'] if stats['total'] > 0 else 0
        lab_matrix[i, j] = pct
        lab_matrix[j, i] = pct

    # Set diagonal to 100% (perfect self-agreement)
    np.fill_diagonal(loc_matrix, 100)
    np.fill_diagonal(lab_matrix, 100)

    # Combine both matrices for clustering (average of both agreement types)
    combined_matrix = (loc_matrix + lab_matrix) / 2

    # Convert similarity to distance and perform hierarchical clustering
    distance_matrix = 100 - combined_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    dendro = dendrogram(linkage_matrix, no_plot=True)

    # Get the order of annotators from clustering
    order = dendro['leaves']
    sorted_annotators = [annotators[i] for i in order]

    # Reorder matrices
    loc_matrix_sorted = loc_matrix[order, :][:, order]
    lab_matrix_sorted = lab_matrix[order, :][:, order]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Location agreement heatmap
    sns.heatmap(
        loc_matrix_sorted,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=sorted_annotators,
        yticklabels=sorted_annotators,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Agreement %'},
        ax=axes[0],
    )
    axes[0].set_title('Location Agreement (%)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Annotator')
    axes[0].set_ylabel('Annotator')

    # Label agreement heatmap
    sns.heatmap(
        lab_matrix_sorted,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=sorted_annotators,
        yticklabels=sorted_annotators,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Agreement %'},
        ax=axes[1],
    )
    axes[1].set_title('Label Agreement (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Annotator')
    axes[1].set_ylabel('Annotator')

    plt.tight_layout()
    plt.savefig(
        f'output/{output_append}/annotator_agreement_heatmaps.png', dpi=300, bbox_inches='tight'
    )


def analyze_label_confusion(annotations, iou_threshold=0.2):
    """
    Analyze which labels are commonly confused between annotators.

    Returns:
        confusion_matrix: Dict of (label1, label2) -> count
        per_annotator_pairs: Dict of annotator pairs -> confusion data
        all_labels: Set of all unique labels
    """
    grouped = parse_annotations(annotations)
    annotators = sorted(set(a[0] for a in annotations))

    # Global confusion matrix
    confusion_matrix = defaultdict(int)

    # Per-annotator-pair confusion
    per_annotator_pairs = defaultdict(lambda: defaultdict(int))

    # Track all labels
    all_labels = set()

    for profile_id, dyes in grouped.items():
        for dye, peaks in dyes.items():
            # Group peaks by annotator
            by_annotator = defaultdict(list)
            for peak in peaks:
                by_annotator[peak['annotator']].append(peak)
                all_labels.add(peak['label'])

            # Compare each pair of annotators
            for ann1, ann2 in itertools.combinations(annotators, 2):
                if ann1 not in by_annotator or ann2 not in by_annotator:
                    continue

                peaks1 = by_annotator[ann1]
                peaks2 = by_annotator[ann2]
                pair = tuple(sorted([ann1, ann2]))

                # Match peaks
                matches = match_peaks_optimal(peaks1, peaks2, iou_threshold)

                # Record label pairs for matched peaks
                for i, j, iou in matches:
                    label1 = peaks1[i]['label']
                    label2 = peaks2[j]['label']

                    # Store in sorted order for consistency
                    label_pair = tuple(sorted([label1, label2]))
                    confusion_matrix[label_pair] += 1
                    per_annotator_pairs[pair][label_pair] += 1

    return confusion_matrix, per_annotator_pairs, all_labels, annotators


def print_confusion_analysis(confusion_matrix, per_annotator_pairs, all_labels, annotators):
    """Print detailed confusion analysis."""
    print('LABEL CONFUSION ANALYSIS')
    print('=' * 70)

    # Overall confusion statistics
    print('\n1. OVERALL LABEL CONFUSION')
    print('-' * 70)

    # Separate agreements from disagreements
    agreements = []
    disagreements = []

    for label_pair, count in confusion_matrix.items():
        if label_pair[0] == label_pair[1]:
            agreements.append((label_pair[0], count))
        else:
            disagreements.append((label_pair, count))

    # Print agreements
    print('\nLabel Agreements (same label assigned):')
    if agreements:
        for label, count in sorted(agreements, key=lambda x: -x[1]):
            print(f'  {label}: {count} times')
    else:
        print('  None found')

    # Print disagreements
    print('\nLabel Disagreements (different labels for same peak):')
    if disagreements:
        for (label1, label2), count in sorted(disagreements, key=lambda x: -x[1]):
            print(f'  {label1} ↔ {label2}: {count} times')
    else:
        print('  None found')

    # Most confused pairs
    if disagreements:
        most_confused = max(disagreements, key=lambda x: x[1])
        print(
            f'\nMost commonly confused: {most_confused[0][0]} ↔ {most_confused[0][1]} ({most_confused[1]} times)'
        )

    # Per-annotator pair analysis
    print('\n\n2. CONFUSION BY ANNOTATOR PAIR')
    print('-' * 70)

    for pair in sorted(per_annotator_pairs.keys()):
        print(f'\n{pair[0]} vs {pair[1]}:')
        confusion = per_annotator_pairs[pair]

        # Separate agreements and disagreements
        pair_agreements = {k: v for k, v in confusion.items() if k[0] == k[1]}
        pair_disagreements = {k: v for k, v in confusion.items() if k[0] != k[1]}

        total = sum(confusion.values())
        agreement_count = sum(pair_agreements.values())
        disagreement_count = sum(pair_disagreements.values())

        print(f'  Agreement: {agreement_count}/{total} ({100 * agreement_count / total:.1f}%)')

        if pair_disagreements:
            print(f'  Disagreements:')
            for (l1, l2), count in sorted(pair_disagreements.items(), key=lambda x: -x[1]):
                pct = 100 * count / total
                print(f'    {l1} ↔ {l2}: {count} times ({pct:.1f}%)')

    # Summary statistics
    print('\n\n3. SUMMARY STATISTICS')
    print('-' * 70)

    total_comparisons = sum(confusion_matrix.values())
    total_agreements = sum(count for (l1, l2), count in confusion_matrix.items() if l1 == l2)
    total_disagreements = sum(count for (l1, l2), count in confusion_matrix.items() if l1 != l2)

    print(f'Total matched peaks compared: {total_comparisons}')
    print(
        f'Label agreements: {total_agreements} ({100 * total_agreements / total_comparisons:.1f}%)'
    )
    print(
        f'Label disagreements: {total_disagreements} ({100 * total_disagreements / total_comparisons:.1f}%)'
    )
    print(f'\nUnique labels found: {len(all_labels)}')
    print(f'Labels: {", ".join(sorted(all_labels))}')


def plot_confusion_matrix(confusion_matrix, all_labels, normalize=True, output_append=''):
    """
    Create a confusion matrix heatmap showing label interchange patterns.

    Args:
        confusion_matrix: Dict of (label1, label2) -> count
        all_labels: Set of all unique labels
        normalize: If True, normalize rows to show percentages (default True)
    """
    labels_sorted = sorted(all_labels)
    n = len(labels_sorted)

    # Build confusion matrix
    matrix = np.zeros((n, n))
    label_to_idx = {label: i for i, label in enumerate(labels_sorted)}

    for (label1, label2), count in confusion_matrix.items():
        i, j = label_to_idx[label1], label_to_idx[label2]
        matrix[i, j] += count
        if i != j:  # Make symmetric for visualization
            matrix[j, i] += count

    # Create figure with two subplots if normalizing
    if normalize:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # First subplot: Raw counts
        ax1 = axes[0]
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            xticklabels=labels_sorted,
            yticklabels=labels_sorted,
            cbar_kws={'label': 'Count'},
            square=True,
            ax=ax1,
        )
        ax1.set_title('Label Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Assigned Label', fontsize=12)
        ax1.set_ylabel('Assigned Label', fontsize=12)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

        # Second subplot: Row-normalized percentages
        ax2 = axes[1]
        matrix_normalized = np.zeros_like(matrix)
        for i in range(n):
            row_sum = matrix[i, :].sum()
            if row_sum > 0:
                matrix_normalized[i, :] = 100 * matrix[i, :] / row_sum

        sns.heatmap(
            matrix_normalized,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=labels_sorted,
            yticklabels=labels_sorted,
            cbar_kws={'label': 'Percentage (%)'},
            square=True,
            vmin=0,
            vmax=100,
            ax=ax2,
        )
        ax2.set_title(
            'Label Confusion Matrix (Row-Normalized)\nEach row shows: When annotators use label Y, what % is confused with X?',
            fontsize=14,
            fontweight='bold',
        )
        ax2.set_xlabel('Confused With Label', fontsize=12)
        ax2.set_ylabel('When Annotator Uses', fontsize=12)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add interpretation text
        fig.text(
            0.5,
            0.02,
            'Left: Raw counts | Right: Row percentages (each row sums to 100%)\n'
            'Diagonal: same label (agreement) | Off-diagonal: different labels (confusion)',
            ha='center',
            fontsize=10,
            style='italic',
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(
            f'output/{output_append}/label_confusion_matrix.png', dpi=300, bbox_inches='tight'
        )
        print("\nConfusion matrix saved to 'label_confusion_matrix.png'")

    else:
        # Original single plot
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            xticklabels=labels_sorted,
            yticklabels=labels_sorted,
            cbar_kws={'label': 'Count'},
            square=True,
        )

        plt.title(
            'Label Confusion Matrix\n(How often label pairs appear together on matched peaks)',
            fontsize=14,
            fontweight='bold',
            pad=20,
        )
        plt.xlabel('Label', fontsize=12)
        plt.ylabel('Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.text(
            0.5,
            -0.15,
            'Diagonal: same label (agreement) | Off-diagonal: different labels (confusion)',
            ha='center',
            transform=plt.gca().transAxes,
            fontsize=10,
            style='italic',
        )

        plt.tight_layout()
        plt.savefig(
            f'output/{output_append}/label_confusion_matrix.png', dpi=300, bbox_inches='tight'
        )
        print("\nConfusion matrix saved to 'label_confusion_matrix.png'")


def plot_confusion_by_annotator(per_annotator_pairs, annotators, output_append=''):
    """Create bar chart showing disagreement rates per annotator pair."""
    pairs = sorted(per_annotator_pairs.keys())

    disagreement_rates = []
    pair_labels = []

    for pair in pairs:
        confusion = per_annotator_pairs[pair]
        total = sum(confusion.values())
        disagreements = sum(count for (l1, l2), count in confusion.items() if l1 != l2)

        if total > 0:
            disagreement_rates.append(100 * disagreements / total)
            pair_labels.append(f'{pair[0]}\nvs\n{pair[1]}')

    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(disagreement_rates)), disagreement_rates, color='coral', edgecolor='darkred'
    )

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, disagreement_rates)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{rate:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    plt.xlabel('Annotator Pair', fontsize=12)
    plt.ylabel('Label Disagreement Rate (%)', fontsize=12)
    plt.title(
        'Label Disagreement Rate by Annotator Pair\n(% of matched peaks with different labels)',
        fontsize=14,
        fontweight='bold',
    )
    plt.xticks(range(len(pair_labels)), pair_labels, fontsize=9)
    plt.ylim(0, max(disagreement_rates) * 1.15 if disagreement_rates else 100)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f'output/{output_append}/disagreement_by_annotator_pair.png', dpi=300, bbox_inches='tight'
    )


def identify_problematic_labels(confusion_matrix, all_labels):
    """Identify labels that are most often confused with others."""
    print('\n\n4. MOST PROBLEMATIC LABELS')
    print('-' * 70)

    # Count how often each label appears in disagreements
    label_confusion_counts = defaultdict(int)

    for (label1, label2), count in confusion_matrix.items():
        if label1 != label2:  # Only disagreements
            label_confusion_counts[label1] += count
            label_confusion_counts[label2] += count

    if label_confusion_counts:
        print('\nLabels most often involved in disagreements:')
        for label, count in sorted(label_confusion_counts.items(), key=lambda x: -x[1]):
            print(f'  {label}: involved in {count} disagreements')
    else:
        print('\nNo disagreements found!')

    # Find which specific pairs are confused
    print('\nSpecific confusion patterns:')
    label_pairs = defaultdict(list)
    for (label1, label2), count in confusion_matrix.items():
        if label1 != label2:
            label_pairs[label1].append((label2, count))
            label_pairs[label2].append((label1, count))

    for label in sorted(label_pairs.keys()):
        partners = sorted(label_pairs[label], key=lambda x: -x[1])
        most_confused_with = partners[0] if partners else None
        if most_confused_with:
            print(
                f"  {label} → most confused with '{most_confused_with[0]}' ({most_confused_with[1]} times)"
            )


def collect_label_flows(annotations, iou_threshold=0.2):
    """
    Collect label flows between each pair of annotators.
    Includes unmatched peaks (false positives/negatives).

    Returns:
        flows: Dict of (ann1, ann2) -> Dict of (label1, label2) -> count
               where label can be None for unmatched peaks
    """
    grouped = parse_annotations(annotations)
    annotators = sorted(set(a[0] for a in annotations))

    flows = defaultdict(lambda: defaultdict(int))

    for profile_id, dyes in grouped.items():
        for dye, peaks in dyes.items():
            # Group peaks by annotator
            by_annotator = defaultdict(list)
            for peak in peaks:
                by_annotator[peak['annotator']].append(peak)

            # Compare each pair of annotators
            for ann1, ann2 in itertools.combinations(annotators, 2):
                if ann1 not in by_annotator or ann2 not in by_annotator:
                    continue

                peaks1 = by_annotator[ann1]
                peaks2 = by_annotator[ann2]
                pair = (ann1, ann2)

                # Match peaks
                matches = match_peaks_optimal(peaks1, peaks2, iou_threshold)

                # Track which peaks were matched
                matched_peaks1 = set()
                matched_peaks2 = set()

                # Record label flows for matched peaks
                for i, j, iou in matches:
                    label1 = peaks1[i]['label']
                    label2 = peaks2[j]['label']
                    flows[pair][(label1, label2)] += 1
                    matched_peaks1.add(i)
                    matched_peaks2.add(j)

                # Record unmatched peaks from ann1 (ann2 missed these)
                for i, peak in enumerate(peaks1):
                    if i not in matched_peaks1:
                        flows[pair][(peak['label'], None)] += 1

                # Record unmatched peaks from ann2 (ann1 missed these)
                for j, peak in enumerate(peaks2):
                    if j not in matched_peaks2:
                        flows[pair][(None, peak['label'])] += 1

    return flows, annotators


def create_sankey_for_pair(ann1, ann2, label_flows, labels_sorted):
    """
    Create a Sankey diagram for a single annotator pair.
    Includes unmatched peaks shown as flows to/from "Baseline" category.

    Args:
        ann1, ann2: Annotator names
        label_flows: Dict of (label1, label2) -> count (label can be None for unmatched)
        all_labels: Set of all unique labels (excluding None)
    """
    if not label_flows:
        return None

    n_labels = len(labels_sorted)

    # Create node labels: [ann1_label1, ann1_label2, ..., ann2_label1, ann2_label2, ...]
    node_labels = [f'{ann1}: {label}' for label in labels_sorted] + [
        f'{ann2}: {label}' for label in labels_sorted
    ]

    # Map labels to indices
    label_to_idx = {label: i for i, label in enumerate(labels_sorted)}

    # Build source, target, value lists for Sankey
    source = []
    target = []
    value = []
    link_colors = []

    for (label1, label2), count in label_flows.items():
        if count > 0:
            # Map None to "Baseline"
            l1 = label1 if label1 is not None else 'Baseline'
            l2 = label2 if label2 is not None else 'Baseline'

            source_idx = label_to_idx[l1]
            target_idx = n_labels + label_to_idx[l2]

            source.append(source_idx)
            target.append(target_idx)
            value.append(count)

            # Color based on flow type
            if l1 == l2 and l1 != 'Baseline':
                color = label_colors.get(l1, 'rgba(100, 100, 100, 0.4)')
            elif l1 == 'Baseline' or l2 == 'Baseline':
                color = 'rgba(230, 126, 34, 0.4)'  # Orange for unmatched
            else:
                color = 'rgba(231, 76, 60, 0.3)'  # Red for disagreements
            link_colors.append(color)

    if not source:
        return None

    # Create Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=node_labels,
                    color=['lightblue'] * n_labels + ['lightcoral'] * n_labels,
                ),
                link=dict(source=source, target=target, value=value, color=link_colors),
            )
        ]
    )

    # Calculate statistics
    total = sum(value)
    matched = sum(v for (l1, l2), v in label_flows.items() if l1 is not None and l2 is not None)
    agreements = sum(v for (l1, l2), v in label_flows.items() if l1 == l2 and l1 is not None)
    unmatched = total - matched
    agreement_pct = 100 * agreements / matched if matched > 0 else 0

    fig.update_layout(
        title=f'Label Flow: {ann1} → {ann2}<br>'
        f'Agreement: {agreement_pct:.1f}% ({agreements}/{matched} matched peaks) | '
        f'Unmatched: {unmatched} peaks',
        font=dict(size=12),
        height=500,
    )

    return fig


def create_all_sankey_diagrams(annotations, iou_threshold=0.2, save_html=True, output_append=''):
    """
    Create Sankey diagrams for all annotator pairs.

    Args:
        annotations: List of annotation tuples
        iou_threshold: Minimum IoU for peak matching
        save_html: If True, save individual HTML files for each pair
    """
    flows, annotators = collect_label_flows(annotations, iou_threshold)
    all_labels = set(a[6] for a in annotations)
    # Add "Baseline" to labels for unmatched peaks
    all_labels = sorted([l for l in all_labels if l is not None]) + ['Baseline']

    if not flows:
        print('No matched peaks found between annotators!')
        return

    print(f'Creating Sankey diagrams for {len(flows)} annotator pairs...\n')

    # Create a grid of Sankey diagrams
    pairs = sorted(flows.keys())
    n_pairs = len(pairs)

    # Calculate grid dimensions
    n_cols = min(2, n_pairs)  # Max 2 columns
    n_rows = (n_pairs + n_cols - 1) // n_cols

    # Create individual diagrams
    for i, (ann1, ann2) in enumerate(pairs, 1):
        label_flows = flows[(ann1, ann2)]
        fig = create_sankey_for_pair(ann1, ann2, label_flows, all_labels)

        if fig:
            if save_html:
                filename = f'output/{output_append}/sankey_{ann1}_vs_{ann2}.html'
                fig.write_html(filename)
                print(f'✓ Saved: {filename}')

    # Create a combined view (simplified)
    create_combined_sankey_view(flows, annotators, all_labels, output_append)

    print(f'\n✓ Created {n_pairs} Sankey diagrams')
    return flows


def create_combined_sankey_view(flows, annotators, all_labels, output_append):
    """
    Create a single Sankey showing all label flows across all annotator pairs.
    """
    # Aggregate flows across all pairs
    aggregated_flows = defaultdict(int)

    for pair_flows in flows.values():
        for (label1, label2), count in pair_flows.items():
            aggregated_flows[(label1, label2)] += count

    if not aggregated_flows:
        return

    labels_sorted = sorted(all_labels)
    n_labels = len(labels_sorted)

    # Create nodes
    node_labels = [f'Assigned: {label}' for label in labels_sorted] + [
        f'Confirmed: {label}' for label in labels_sorted
    ]

    label_to_idx = {label: i for i, label in enumerate(labels_sorted)}

    # Build links
    source = []
    target = []
    value = []
    link_colors = []

    for (label1, label2), count in aggregated_flows.items():
        if count > 0:
            # Map None to "Baseline"
            label1 = label1 if label1 is not None else 'Baseline'
            label2 = label2 if label2 is not None else 'Baseline'

            source_idx = label_to_idx[label1]
            target_idx = n_labels + label_to_idx[label2]

            source.append(source_idx)
            target.append(target_idx)
            value.append(count)

            if label1 == label2:
                color = label_colors.get(label1, 'rgba(100, 100, 100, 0.5)')
            else:
                color = 'rgba(231, 76, 60, 0.4)'
            link_colors.append(color)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=25,
                    line=dict(color='black', width=0.5),
                    label=node_labels,
                    color=['lightblue'] * n_labels + ['lightcoral'] * n_labels,
                ),
                link=dict(source=source, target=target, value=value, color=link_colors),
            )
        ]
    )

    total = sum(value)
    agreements = sum(v for (l1, l2), v in aggregated_flows.items() if l1 == l2)
    agreement_pct = 100 * agreements / total if total > 0 else 0

    fig.update_layout(
        title=f'Combined Label Flow: All Annotator Pairs<br>'
        f'Overall Agreement: {agreement_pct:.1f}% ({agreements}/{total} matched peaks)',
        font=dict(size=14),
        height=600,
        width=1000,
    )

    fig.write_html(f'output/{output_append}/sankey_combined.html')
    print('\n✓ Saved combined view: sankey_combined.html')


def print_flow_summary(flows, annotators):
    """Print summary statistics for each annotator pair."""
    print('\nLABEL FLOW SUMMARY')
    print('=' * 70)

    for (ann1, ann2), label_flows in sorted(flows.items()):
        total = sum(label_flows.values())
        agreements = sum(count for (l1, l2), count in label_flows.items() if l1 == l2)
        disagreements = total - agreements

        print(f'\n{ann1} vs {ann2}:')
        print(f'  Total matched peaks: {total}')
        print(f'  Agreements: {agreements} ({100 * agreements / total:.1f}%)')
        print(f'  Disagreements: {disagreements} ({100 * disagreements / total:.1f}%)')

        # Show top disagreement patterns
        disagree_flows = [
            (labels, count) for labels, count in label_flows.items() if labels[0] != labels[1]
        ]
        if disagree_flows:
            print(f'  Top disagreements:')
            for (l1, l2), count in sorted(disagree_flows, key=lambda x: -x[1])[:3]:
                print(f'    {l1} → {l2}: {count} times')


def compare_annotators(annotations: Iterable, output_append: str = '', bins=[[-100, 1000000]]):
    """
    Compares the annotators that have annotated the same profiles from the annotations in a csv file
    """

    # Run analysis
    print('Using optimal matching (Hungarian algorithm) with IoU threshold = 0.2\n')
    for bin in bins:
        os.makedirs(f'output/{output_append + str(bin[0])}', exist_ok=True)

        bin_annotations = [ann for ann in annotations if bin[0] < ann[5] < bin[1]]
        loc_agree, lab_agree, annotators = compute_pairwise_agreement(
            bin_annotations, iou_threshold=0.2, use_optimal_matching=True
        )
        print_results(loc_agree, lab_agree, annotators)
        plot_agreement_heatmaps(
            loc_agree, lab_agree, annotators, output_append=output_append + str(bin[0])
        )

        # Run analysis
        print('Analyzing label confusion patterns...\n')
        confusion_matrix, per_annotator_pairs, all_labels, annotators = analyze_label_confusion(
            bin_annotations, iou_threshold=0.2
        )

        # print_confusion_analysis(confusion_matrix, per_annotator_pairs, all_labels, annotators)
        # identify_problematic_labels(confusion_matrix, all_labels)

        # Create visualizations
        if len(all_labels) > 1:
            plot_confusion_matrix(
                confusion_matrix, all_labels, output_append=output_append + str(bin[0])
            )
            plot_confusion_by_annotator(
                per_annotator_pairs, annotators, output_append=output_append + str(bin[0])
            )
        else:
            print('\nNot enough label variety to create confusion matrix visualization')

        print('Creating Sankey diagrams for annotator label flows...\n')
        flows = create_all_sankey_diagrams(
            bin_annotations,
            iou_threshold=0.2,
            save_html=True,
            output_append=output_append + str(bin[0]),
        )
        # print_flow_summary(flows, sorted(set(a[0] for a in annotations)))


def run(annotations_folder_path: str, data_config: str, output_file: str):
    # also do a comparison for profiles with multiple annotators
    # first, bring all separate annotation files together into one file (to remove duplicates)
    per_annotator = True
    sample_arrays = parse_full_annotations(annotations_folder_path, per_annotator=per_annotator)

    profile_data = load_dataset(data_config)
    # called_alleles = parse_full_annotations_to_called_alleles(folder,
    #                                                           profile_data,
    #                                                           per_annotator=per_annotator)
    write_predictions_to_csv(
        sample_arrays, output_file, per_annotator=per_annotator, hidimages=profile_data
    )
    annotations = parse_csv_file(output_file)
    # filter on our annotators

    compare_annotators(annotations)
    # compare per bin
    # compare_annotators(annotations, bins=[[-10,100], [100, 800,], [800, 100000]])

    # also per ref/trace
    for start in ['A', 'W']:
        os.makedirs(f'output/{start}', exist_ok=True)
        profile_annotations = [a for a in annotations if a[1].split('_')[2][0] == start]
        n_annotators = len({a[0] for a in profile_annotations})
        print({a[1] for a in profile_annotations}, n_annotators)
        if n_annotators > 1:
            compare_annotators(
                [a for a in annotations if a[1].split('_')[2][0] == start], output_append=start
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a',
        '--annotation-folder-path',
        type=str,
        required=True,
        help='path to the folder containing the annotation csv files',
    )
    parser.add_argument(
        '-d',
        '--data-config',
        type=str,
        required=True,
        help='The path to a .yaml file (or the basename without an extension '
        'if located in `./config/data/`) containing the configuration '
        'for the dataset',
    )
    parser.add_argument(
        '-o',
        '--output-file',
        type=str,
        required=True,
        default='output/annotations_repeated.csv',
        help='path to the output csv file where combined annotations will be saved',
    )

    args = parser.parse_args()
    run(args.annotation_folder_path, args.data_config, args.output_file)
