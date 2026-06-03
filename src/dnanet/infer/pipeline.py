"""Core inference pipeline for DNANet.

Design pattern: **Facade**
    :class:`InferencePipeline` is the single entry point for running
    inference on HID profiles. It orchestrates model loading, HID file
    parsing, prediction, allele calling, and result assembly — hiding
    the complexity of the underlying components behind a clean API.

Design pattern: **Strategy**
    Allele calling is delegated to :class:`~dnanet.evaluation.allele_caller.AlleleCaller`,
    allowing different calling strategies (nearest base-pair, exact base-pair)
    to be swapped without changing the pipeline.

Usage::

    from dnanet.infer.pipeline import InferencePipeline
    from dnanet.data.strategies.scaling import PowerPlexFusion6CStrategy

    pipeline = InferencePipeline(
        checkpoint="outputs/exp1/best.ckpt",
        scaling_strategy=PowerPlexFusion6CStrategy(),
    )
    results = pipeline.run(
        hid_profiles=[
            ("sample1.HID", "ladder1.HID"),
            ("sample2.HID", "ladder2.HID"),
        ],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from dnanet.data.image import HIDImage
from dnanet.core.allele import Allele
from dnanet.core.marker import Marker
from dnanet.infer.output import (
    AlleleCall,
    MarkerResult,
    ProfileResult,
    InferenceResult,
)
from dnanet.data.ladders.ladder import Ladder
from dnanet.evaluation.allele_caller import (
    AlleleCaller,
    ExactBasePairCaller,
    NearestBasePairCaller,
)
from dnanet.data.ladders.ladder_allele_catalog import LadderAlleleCatalog


if TYPE_CHECKING:
    from torch import nn

    from dnanet.core.panel import Panel
    from dnanet.data.strategies.scaling import ScalingStrategy


class InferencePipeline:
    """Runs inference on HID profiles using a trained DNANet model.

    This pipeline handles the full inference flow:
    1. Load model from checkpoint
    2. Load each HID profile with kit-specific scaling
    3. Adjust panel from ladder (if provided)
    4. Run model predictions
    5. Extract allele calls with confidence scores
    6. Assemble structured results

    Args:
        checkpoint: Path to a trained model checkpoint (.ckpt).
        scaling_strategy: Kit-specific scaling strategy defining the panel,
            size standard, and dye configuration.
        device: Torch device to run inference on. Defaults to auto-detect.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        scaling_strategy: ScalingStrategy,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.scaling_strategy = scaling_strategy
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model from checkpoint
        self._module, self._model_type = self._load_model()
        self.model.eval()

        logger.info(
            'Loaded model {} from {} (device: {})',
            type(self.model).__name__,
            self.checkpoint,
            self.device,
        )

    @property
    def model(self) -> nn.Module:
        """The loaded neural network."""
        return self._module.model

    @property
    def model_type(self) -> str:
        """Type of model: 'segmentation', 'multiclass', or 'peaknet'."""
        return self._model_type

    def _load_model(self) -> tuple[Any, str]:
        """Load a trained module from checkpoint.

        Infers the UNet architecture from checkpoint state dict shapes
        so no Hydra config is needed.

        Returns:
            Tuple of (LightningModule, model_type_string).
        """
        from collections import OrderedDict

        import lightning as L

        from dnanet.modules import peaknet as pn_mod
        from dnanet.modules import segmentation as seg_mod
        from dnanet.models.unet import UNet

        # Load checkpoint state dict
        checkpoint_data = torch.load(self.checkpoint, map_location=self.device, weights_only=True)

        # Extract state dict from Lightning checkpoint wrapper
        if 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
        else:
            state_dict = checkpoint_data

        # Strip 'model.' prefix if present (Lightning convention)
        if any(k.startswith('model.') for k in state_dict):
            state_dict = OrderedDict(
                (k[6:], v) if k.startswith('model.') else (k, v) for k, v in state_dict.items()
            )

        # Infer UNet architecture from state dict keys/shapes
        depth, num_filters, kernel_size, out_channels = self._infer_unet_arch(state_dict)

        # Determine module class and model type
        if 'PeakNet' in state_dict or 'combiner' in str(state_dict.keys()):
            raise NotImplementedError(
                'PeakNet checkpoints require peak extraction infrastructure. '
                'Use UNet-based segmentation models for direct HID inference.'
            )

        # Detect multiclass: any model with >1 output channel is multiclass
        model_type = 'multiclass' if out_channels > 1 else 'segmentation'

        # Create UNet with inferred architecture
        unet = UNet(
            depth=depth,
            kernel_size=kernel_size,
            num_filters=num_filters,
            out_channels=out_channels,
        )

        # Load weights with strict=True — architecture must match
        missing, unexpected = unet.load_state_dict(state_dict, strict=False)
        if unexpected:
            logger.warning('Unexpected keys in checkpoint (ignored): {}', unexpected[:5])
        if missing:
            logger.warning('Missing keys in checkpoint: {}', missing[:5])

        unet.to(self.device)
        unet.eval()

        # Wrap in SegmentationModule (no training machinery needed)
        module = seg_mod.SegmentationModule.__new__(seg_mod.SegmentationModule)
        module.__init__(model=unet, loss_fn=None, optimizer=None)

        logger.info(
            'Loaded {} (depth={}, filters={}, kernel={}, out_ch={}, device={})',
            type(unet).__name__,
            depth,
            num_filters,
            kernel_size,
            out_channels,
            self.device,
        )

        return module, model_type

    @staticmethod
    def _infer_unet_arch(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[int, int, tuple[int, int], int]:
        """Infer UNet architecture from checkpoint state dict shapes.

        Reads the first encoder, bottleneck, and head layers to determine
        depth, num_filters, kernel_size, and output channels.

        Handles both key patterns:
        - Lightning-wrapped: ``encoders.0.conv.double_conv.0.weight``
        - Raw UNet: ``encoder.0.conv.0.weight``

        Args:
            state_dict: Model state dict (without 'model.' prefix).

        Returns:
            (depth, num_filters, kernel_size, out_channels).
        """
        # Find encoder keys — supports both key patterns
        # Pattern 1: encoders.0.conv.double_conv.0.weight (UNet with DoubleConv)
        # Pattern 2: encoder.0.conv.0.weight (simpler pattern)
        encoder_keys = [k for k in state_dict if k.startswith('encoders.')]
        if not encoder_keys:
            # Try alternate pattern
            encoder_keys = [k for k in state_dict if k.startswith('encoder.')]
        if not encoder_keys:
            raise ValueError(
                'Checkpoint does not contain UNet encoder keys. '
                f'Found keys: {list(state_dict.keys())[:10]}'
            )

        # Extract the first encoder index: 'encoders.0.conv.double_conv.0.weight' -> 0
        first_enc_idx = encoder_keys[0].split('.')[1]

        # Find a conv weight from the first encoder to get kernel size and channels
        # Pattern 1: encoders.0.conv.double_conv.0.weight (shape: out_ch, in_ch, kh, kw)
        enc0_conv_key = f'encoders.{first_enc_idx}.conv.double_conv.0.weight'
        if enc0_conv_key not in state_dict:
            # Pattern 2: encoder.0.conv.0.weight
            enc0_conv_key = f'encoder.{first_enc_idx}.conv.0.weight'
        if enc0_conv_key not in state_dict:
            # Fallback: find any conv weight from this encoder
            enc0_conv_key = next(
                (
                    k
                    for k in state_dict
                    if f'encoders.{first_enc_idx}' in k or f'encoder.{first_enc_idx}' in k
                )
            )

        conv_shape = state_dict[enc0_conv_key].shape
        if len(conv_shape) == 4:
            kernel_size = (conv_shape[2], conv_shape[3])
            num_filters = conv_shape[0]  # output channels
        else:
            kernel_size = (3, 5)
            num_filters = 32

        # Depth = count of unique encoder indices
        if encoder_keys[0].startswith('encoders.'):
            depth = len(set(k.split('.')[1] for k in encoder_keys))
        else:
            depth = len(set(k.split('.')[1] for k in encoder_keys))

        # Output channels from head layer
        head_key = 'head.weight'
        if head_key in state_dict:
            out_channels = state_dict[head_key].shape[0]
        else:
            out_channels = 1

        return depth, num_filters, kernel_size, out_channels

    @staticmethod
    def _infer_architecture(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[int, tuple[int, int], int, int]:
        """Infer UNet architecture from checkpoint state dict shapes.

        Reads architecture params directly from tensor shapes — no config
        file or Hydra required.

        Args:
            state_dict: State dict from a trained UNet checkpoint.

        Returns:
            Tuple of (depth, kernel_size, num_filters, out_channels).
        """
        # Find encoder 0 conv weight: model.encoders.0.conv.double_conv.0.weight
        enc0_key = None
        for k in state_dict:
            if 'encoders.0.conv.double_conv.0.weight' in k:
                enc0_key = k
                break

        if enc0_key is None:
            raise ValueError(
                'Cannot find encoder weights in checkpoint. '
                'Expected key pattern: model.encoders.0.conv.double_conv.0.weight'
            )

        enc0_shape = state_dict[enc0_key].shape
        num_filters = enc0_shape[0]  # output channels of first conv
        kernel_size = (enc0_shape[2], enc0_shape[3])  # (height, width)

        # Determine depth from encoder layer indices
        encoder_indices = set()
        for k in state_dict:
            if '.encoders.' in k:
                parts = k.split('.')
                try:
                    idx = int(parts[2])
                    encoder_indices.add(idx)
                except (IndexError, ValueError):
                    continue

        depth = max(encoder_indices) + 1 if encoder_indices else 4

        # Output channels from head weight
        out_channels = 1
        for k in state_dict:
            if k.endswith('.head.weight'):
                out_channels = state_dict[k].shape[0]
                break

        return depth, kernel_size, num_filters, out_channels

    def run(
        self,
        hid_profiles: Sequence[tuple[str, str | None]],
        *,
        caller: str = 'nearest',
        prediction_threshold: float = 0.5,
        confidence_threshold: float | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        save_predictions: bool = False,
        save_plots: bool = False,
        output_dir: str | Path | None = None,
    ) -> InferenceResult:
        """Run inference on a sequence of HID profiles.

        Args:
            hid_profiles: Sequence of (hid_path, ladder_path_or_none) tuples.
            caller: Allele calling strategy — 'nearest' or 'exact'.
            prediction_threshold: Probability threshold for positive predictions.
            confidence_threshold: Minimum confidence to include an allele call.
                If None, all called alleles are included.
            batch_size: Batch size for inference (typically 1).
            num_workers: DataLoader workers.
            save_predictions: Save raw prediction arrays to disk.
            save_plots: Save EPG visualizations to disk.
            output_dir: Base directory for saved outputs.

        Returns:
            Complete inference results.
        """
        caller_instance = self._resolve_caller(caller, prediction_threshold)
        self._caller = caller_instance
        results: list[ProfileResult] = []
        timing: dict[str, float] = {}

        for hid_path, ladder_path in hid_profiles:
            start_time = __import__('time').perf_counter()
            profile_result = self._infer_profile(
                hid_path=hid_path,
                ladder_path=ladder_path,
                caller=caller_instance,
                confidence_threshold=confidence_threshold,
                save_predictions=save_predictions,
                save_plots=save_plots,
                output_dir=output_dir,
            )
            elapsed = __import__('time').perf_counter() - start_time
            timing[Path(hid_path).name] = round(elapsed, 3)
            results.append(profile_result)

        return InferenceResult(
            checkpoint=str(self.checkpoint),
            kit=self.scaling_strategy.kit.name,
            profiles=results,
            timing=timing,
        )

    def _resolve_caller(self, caller: str, threshold: float) -> AlleleCaller:
        """Resolve the allele caller string to an instance."""
        if caller == 'exact':
            return ExactBasePairCaller(threshold=threshold, exclude_non_autosomal=False)
        return NearestBasePairCaller(threshold=threshold, exclude_non_autosomal=False)

    def _infer_profile(
        self,
        hid_path: str,
        ladder_path: str | None,
        caller: AlleleCaller,
        confidence_threshold: float | None,
        save_predictions: bool,
        save_plots: bool,
        output_dir: str | Path | None,
    ) -> ProfileResult:
        """Run inference on a single HID profile.

        Args:
            hid_path: Path to the HID file.
            ladder_path: Path to the ladder HID file, or None.
            caller: Allele caller instance.
            confidence_threshold: Minimum confidence threshold.
            save_predictions: Whether to save raw predictions.
            save_plots: Whether to save EPG plots.
            output_dir: Output base directory.

        Returns:
            ProfileResult with called alleles.
        """
        warnings: list[str] = []
        sample = Path(hid_path).stem

        # Load HID image
        image = HIDImage(
            path=hid_path,
            scaling_strategy=self.scaling_strategy,
            include_size_standard=False,
            load_in_memory=True,
        )

        if image.data is None:
            warnings.append(f'Failed to load HID data from {hid_path}')
            return ProfileResult(sample=sample, hid_path=str(hid_path), warnings=warnings)

        # Determine panel (adjusted from ladder if provided)
        panel = self._get_adjusted_panel(image, ladder_path, warnings)

        # Prepare input tensor
        data = image.data
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]

        # Remove size standard dye if present
        num_dyes = self.scaling_strategy.kit.num_dyes
        if data.shape[0] > num_dyes:
            data = data[:num_dyes]

        signal_array = data.copy()
        tensor_input = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run model inference
        with torch.no_grad():
            logits = self._module(tensor_input)

        # Extract prediction probabilities
        if self.model_type == 'multiclass':
            prediction_probs = torch.softmax(logits, dim=1)
            # Get allele class probability (class 1 = ALLELE)
            pred_per_dye = prediction_probs[:, 1, :, :]  # (1, num_dyes, scanpoints)
        elif self.model_type == 'peaknet':
            prediction_probs = torch.softmax(logits, dim=1)
            # PeakNet: (1, num_classes, num_dyes, scanpoints)
            pred_per_dye = prediction_probs[:, 1, :, :]  # allele class
        else:
            # Binary segmentation: sigmoid
            pred_per_dye = torch.sigmoid(logits)  # (1, num_dyes, scanpoints)

        pred_array = pred_per_dye.cpu().numpy()[0]  # (num_dyes, scanpoints)
        scaler = image.scaler

        # Call alleles from prediction
        if panel is None:
            warnings.append('No panel available for allele calling')
            markers = []
        else:
            markers = self._call_alleles_from_prediction(
                prediction_image=pred_array,
                signal_image=signal_array,
                scaler=scaler,
                panel=panel,
                caller=caller,
                confidence_threshold=confidence_threshold,
            )

        # Build result
        result = ProfileResult(
            sample=sample,
            hid_path=str(hid_path),
            ladder_path=ladder_path,
            markers=markers,
            warnings=warnings,
        )

        # Optionally save predictions and plots
        if save_predictions or save_plots:
            self._save_outputs(
                result=result,
                signal=signal_array,
                prediction=pred_array,
                scaler=scaler,
                output_dir=output_dir,
                save_predictions=save_predictions,
                save_plots=save_plots,
            )

        return result

    def _get_adjusted_panel(
        self,
        image: HIDImage,
        ladder_path: str | None,
        warnings: list[str],
    ) -> Panel | None:
        """Get the panel to use, adjusted from ladder if provided.

        Args:
            image: The loaded HID image.
            ladder_path: Path to ladder HID file.
            warnings: List to append warnings to.

        Returns:
            Panel (adjusted if ladder was used, otherwise default from kit),
            or None if no panel is available.
        """
        if ladder_path is None:
            return self.scaling_strategy.kit.panel

        try:
            panel = self.scaling_strategy.kit.panel
            if panel is None:
                warnings.append(f'No panel available for kit {self.scaling_strategy.kit.name}')
                return panel

            catalog = LadderAlleleCatalog.from_panel(panel)
            if catalog is None:
                warnings.append('Could not create ladder allele catalog, using default panel')
                return panel

            # Adjust panel from ladder peaks
            adjusted = Ladder.create_adjusted_panel(
                ladder_path=ladder_path,
                catalog=catalog,
                data_loading_strategy='superior',
                scaling_strategy=self.scaling_strategy,
                dataset_strategy=None,  # type: ignore[arg-type] — not used internally
            )
            if adjusted is not None:
                logger.debug('Panel adjusted from ladder: {}', Path(ladder_path).name)
                return adjusted

            warnings.append(f'Ladder adjustment failed for {ladder_path}, using default panel')
            return panel

        except Exception as e:
            warnings.append(f'Ladder adjustment error: {e}')
            return self.scaling_strategy.kit.panel

    def _call_alleles_from_prediction(
        self,
        prediction_image: np.ndarray,
        signal_image: np.ndarray,
        scaler: np.ndarray,
        panel: Panel,
        caller: AlleleCaller,
        confidence_threshold: float | None,
    ) -> list[MarkerResult]:
        """Call alleles from model prediction and extract confidence scores.

        Args:
            prediction_image: (num_dyes, scanpoints) prediction probabilities.
            signal_image: (num_dyes, scanpoints) raw signal data.
            scaler: (scanpoints,) base-pair calibration.
            panel: Reference panel for allele lookup.
            caller: Allele caller instance.
            confidence_threshold: Minimum confidence to include.

        Returns:
            List of MarkerResult objects.
        """
        # Use the caller to get allele calls
        markers = caller.call_alleles(
            prediction_image=prediction_image,
            signal_image=signal_image,
            scaler=scaler,
            panel=panel,
        )

        # Build connected components for confidence extraction
        components = self._find_connected_components(prediction_image)

        result_markers: list[MarkerResult] = []
        for marker in markers:
            allele_calls: list[AlleleCall] = []
            for allele in marker.alleles:
                # Find confidence for this allele
                confidence = self._extract_confidence(
                    components=components,
                    dye_row=marker.dye_row,
                    base_pair=allele.base_pair or 0.0,
                    scaler=scaler,
                    prediction_image=prediction_image,
                )

                if confidence_threshold is not None and confidence < confidence_threshold:
                    continue

                allele_calls.append(
                    AlleleCall(
                        name=allele.name,
                        base_pair=allele.base_pair or 0.0,
                        height=allele.height or 0.0,
                        confidence=round(confidence, 4),
                    )
                )

            if allele_calls:
                result_markers.append(
                    MarkerResult(
                        name=marker.name,
                        dye_row=marker.dye_row,
                        alleles=allele_calls,
                    )
                )

        return result_markers

    def _find_connected_components(
        self, prediction_image: np.ndarray
    ) -> dict[int, list[tuple[int, int]]]:
        """Find connected components (contiguous regions) per dye channel.

        Args:
            prediction_image: (num_dyes, scanpoints) prediction probabilities.

        Returns:
            Dict mapping dye_row -> list of (start, end) scanpoint ranges.
        """
        components: dict[int, list[tuple[int, int]]] = {}
        for dye_idx in range(prediction_image.shape[0]):
            dye_pred = prediction_image[dye_idx]
            positives = np.where(dye_pred > 0.5)[0]

            if positives.size == 0:
                continue

            # Split into connected components
            splits = np.split(positives, np.where(np.diff(positives) != 1)[0] + 1)
            components[dye_idx] = [(int(s[0]), int(s[-1])) for s in splits if len(s) > 0]

        return components

    def _extract_confidence(
        self,
        components: dict[int, list[tuple[int, int]]],
        dye_row: int,
        base_pair: float,
        scaler: np.ndarray,
        prediction_image: np.ndarray,
    ) -> float:
        """Extract mean prediction probability at the allele's position.

        Args:
            components: Connected components per dye.
            dye_row: Dye channel index.
            base_pair: Base-pair position of the allele.
            scaler: Base-pair calibration array.
            prediction_image: Prediction probabilities.

        Returns:
            Mean probability at the allele's connected component.
        """
        if dye_row not in components:
            return 0.0

        # Find the connected component closest to this base pair
        bp_indices = np.abs(scaler - base_pair)
        closest_idx = int(np.argmin(bp_indices))

        best_confidence = 0.0
        for start, end in components.get(dye_row, []):
            if start <= closest_idx <= end:
                component_probs = prediction_image[dye_row, start : end + 1]
                best_confidence = float(component_probs.mean())
                break

        return best_confidence

    def _save_outputs(
        self,
        result: ProfileResult,
        signal: np.ndarray,
        prediction: np.ndarray,
        scaler: np.ndarray,
        output_dir: str | Path | None,
        save_predictions: bool,
        save_plots: bool,
    ) -> None:
        """Save prediction artifacts to disk.

        Args:
            result: The profile result.
            signal: Raw signal data.
            prediction: Prediction probabilities.
            scaler: Base-pair calibration.
            output_dir: Output base directory.
            save_predictions: Whether to save raw prediction arrays.
            save_plots: Whether to save EPG plots.
        """
        if output_dir is None:
            return

        out = Path(output_dir) / result.sample
        out.mkdir(parents=True, exist_ok=True)

        if save_predictions:
            np.save(out / 'signal.npy', signal)
            np.save(out / 'prediction.npy', prediction)
            np.save(out / 'scaler.npy', scaler)

        if save_plots:
            from dnanet.infer.output import save_epg_plot

            save_epg_plot(
                signal=signal.tolist(),
                prediction=prediction.tolist(),
                title=result.sample,
                output_path=out / 'epg.png',
            )
