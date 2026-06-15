"""DNANet inference — run allele calling on HID profiles.

This module provides a clean programmatic API for running inference
on forensic DNA profiles using trained DNANet models.

Quick start::

    from dnanet.infer import DNANetInfer
    from dnanet.data.strategies.scaling import PowerPlexFusion6CStrategy

    # Single profile
    result = DNANetInfer.run(
        checkpoint="outputs/exp1/best.ckpt",
        hid_profiles=[("sample1.HID", "ladder1.HID")],
        scaling_strategy=PowerPlexFusion6CStrategy(),
    )
    print(result.to_dict()['profiles'][0]['markers'])

    # Batch with options
    result = DNANetInfer.run(
        checkpoint="outputs/exp1/best.ckpt",
        hid_profiles=[
            ("sample1.HID", "ladder1.HID"),
            ("sample2.HID", "ladder1.HID"),
        ],
        scaling_strategy=PowerPlexFusion6CStrategy(),
        output_dir="inference_results/",
        save_plots=True,
        confidence_threshold=0.3,
    )

Kit switching::

    from dnanet.data.strategies.scaling import GlobalFilerStrategy

    result = DNANetInfer.run(
        checkpoint="...",
        hid_profiles=[...],
        scaling_strategy=GlobalFilerStrategy(),  # different kit
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from dnanet.infer.output import AlleleCall, MarkerResult, ProfileResult, InferenceResult
from dnanet.infer.pipeline import InferencePipeline


if TYPE_CHECKING:
    from pathlib import Path

    from dnanet.data.strategies.scaling import ScalingStrategy


class DNANetInfer:
    """High-level inference API for DNANet.

    This is the main entry point for running inference. It provides
    both a classmethod ``run()`` for simple one-off inference and
    an instance-based API for more control.

    Example (classmethod)::

        result = DNANetInfer.run(
            checkpoint="best.ckpt",
            hid_profiles=[("sample.HID", "ladder.HID")],
            scaling_strategy=PowerPlexFusion6CStrategy(),
        )

    Example (instance)::

        infer = DNANetInfer(
            checkpoint="best.ckpt",
            scaling_strategy=PowerPlexFusion6CStrategy(),
        )
        result1 = infer.run_profiles([("sample1.HID", "ladder.HID")])
        result2 = infer.run_profiles([("sample2.HID", "ladder.HID")])
    """

    @staticmethod
    def run(
        checkpoint: str | Path,
        hid_profiles: Sequence[tuple[str, str | None]],
        scaling_strategy: ScalingStrategy,
        *,
        caller: str = 'nearest',
        prediction_threshold: float = 0.5,
        confidence_threshold: float | None = None,
        save_predictions: bool = False,
        save_plots: bool = False,
        output_dir: str | Path | None = None,
        device: str | None = None,
    ) -> InferenceResult:
        """Run inference on HID profiles.

        This is the primary entry point. It creates an
        :class:`InferencePipeline` internally and runs all profiles.

        Args:
            checkpoint: Path to a trained model checkpoint (.ckpt).
            hid_profiles: Sequence of (hid_path, ladder_path_or_none) tuples.
            scaling_strategy: Kit-specific scaling strategy.
            caller: Allele calling strategy — 'nearest' or 'exact'.
            prediction_threshold: Probability threshold for allele calling.
            confidence_threshold: Minimum confidence to include an allele.
            save_predictions: Save raw prediction arrays to disk.
            save_plots: Save EPG visualizations to disk.
            output_dir: Base directory for saved outputs.
            device: Torch device ('cuda', 'cpu', or None for auto-detect).

        Returns:
            Complete inference results.

        Raises:
            FileNotFoundError: If checkpoint or config is missing.
            ValueError: If checkpoint format is invalid.
        """
        pipeline = InferencePipeline(
            checkpoint=checkpoint,
            scaling_strategy=scaling_strategy,
            device=device,
        )
        return pipeline.run(
            hid_profiles=hid_profiles,
            caller=caller,
            prediction_threshold=prediction_threshold,
            confidence_threshold=confidence_threshold,
            save_predictions=save_predictions,
            save_plots=save_plots,
            output_dir=output_dir,
        )


__all__ = [
    'DNANetInfer',
    'InferencePipeline',
    'InferenceResult',
    'ProfileResult',
    'MarkerResult',
    'AlleleCall',
]
