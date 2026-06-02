"""Inference task — run allele calling on HID profiles from CLI.

Design pattern: **Facade**
    Single entry point for CLI-based inference. Reads the composed Hydra
    config and runs the inference pipeline on HID profiles.

Usage::

    dnanet task=infer checkpoint=/path/to/best.ckpt hid_profiles='["sample1.HID"]'
    dnanet task=infer checkpoint=/path/to/best.ckpt kit=PPF6C save_plots=true
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Sequence
from pathlib import Path

from loguru import logger

from dnanet.infer import DNANetInfer


if TYPE_CHECKING:
    from omegaconf import DictConfig

    from dnanet.data.strategies.scaling import ScalingStrategy


_KIT_TO_STRATEGY = {
    'PPF6C': 'powerplex_fusion_6c',
    'GF': 'globalfiler',
    'PY23': 'powerplex_y23',
}


_STRATEGY_CLASSES = {
    'powerplex_fusion_6c': 'dnanet.data.strategies.scaling.PowerPlexFusion6CStrategy',
    'globalfiler': 'dnanet.data.strategies.scaling.GlobalFilerStrategy',
    'powerplex_y23': 'dnanet.data.strategies.scaling.PowerplexY23',
}


def _resolve_strategy(cfg: DictConfig) -> 'ScalingStrategy':
    """Resolve the scaling strategy from Hydra config."""
    from hydra.utils import get_class

    kit = cfg.get('kit')
    if kit:
        strategy_name = _KIT_TO_STRATEGY.get(kit)
        if not strategy_name:
            raise ValueError(f"Unknown kit: '{kit}'. Supported: {list(_KIT_TO_STRATEGY.keys())}")
        logger.info('Creating scaling strategy from kit: {}', kit)
        cls = get_class(_STRATEGY_CLASSES[strategy_name])
        return cls()

    strategy_name = cfg.get('scaling_strategy')
    if strategy_name:
        logger.info('Creating scaling strategy from config: {}', strategy_name)
        cls = get_class(_STRATEGY_CLASSES.get(strategy_name, strategy_name))
        return cls()

    logger.info('Using default strategy: powerplex_fusion_6c')
    cls = get_class(_STRATEGY_CLASSES['powerplex_fusion_6c'])
    return cls()


def _parse_hid_profiles(cfg: DictConfig) -> Sequence[tuple[str, str | None]]:
    """Parse hid_profiles from Hydra config.

    Supports:
    - JSON string: '[["sample1.HID", "ladder1.HID"]]'
    - Single string: "sample1.HID" (no ladder)
    - List in config: hid_profiles: [sample1.HID, sample2.HID]
    """
    profiles = cfg.get('hid_profiles')
    if not profiles:
        return []

    if isinstance(profiles, str):
        try:
            parsed = json.loads(profiles)
            if isinstance(parsed, list):
                return [(p[0], p[1]) if len(p) > 1 else (p[0], None) for p in parsed]
            return [(parsed, None)]
        except json.JSONDecodeError:
            return [(profiles, None)]

    if isinstance(profiles, (list, tuple)):
        result = []
        for p in profiles:
            if isinstance(p, (list, tuple)) and len(p) >= 1:
                result.append((str(p[0]), str(p[1]) if len(p) > 1 else None))
            else:
                result.append((str(p), None))
        return result

    return [(str(profiles), None)]


def run(cfg: DictConfig) -> None:
    """Run inference from Hydra config.

    Args:
        cfg: Composed Hydra config. Must include ``checkpoint`` path.
    """
    checkpoint_path = cfg.get('checkpoint')
    if not checkpoint_path:
        raise ValueError(
            'Inference requires a checkpoint path. '
            'Set it via: dnanet task=infer checkpoint=/path/to/model.ckpt'
        )

    logger.info('Loading checkpoint: {}', checkpoint_path)
    logger.info('Running inference on HID profiles...')

    scaling_strategy = _resolve_strategy(cfg)
    hid_profiles = _parse_hid_profiles(cfg)

    if not hid_profiles:
        logger.warning('No HID profiles specified. Nothing to do.')
        return

    logger.info('Scaling strategy: {}', type(scaling_strategy).__name__)
    logger.info('HID profiles: {}', len(hid_profiles))

    output_dir = cfg.get('output_dir', None)
    if output_dir and cfg.get('save_json', True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    result = DNANetInfer.run(
        checkpoint=checkpoint_path,
        hid_profiles=hid_profiles,
        scaling_strategy=scaling_strategy,
        caller=cfg.get('caller', 'nearest'),
        prediction_threshold=cfg.get('prediction_threshold', 0.5),
        confidence_threshold=cfg.get('confidence_threshold', None),
        batch_size=cfg.get('batch_size', 1),
        num_workers=cfg.get('num_workers', 0),
        save_predictions=cfg.get('save_predictions', False),
        save_plots=cfg.get('save_plots', False),
        output_dir=output_dir,
        device=cfg.get('device', None),
    )

    # Save results
    if output_dir and cfg.get('save_json', True):
        json_path = result.save_json(output_dir / 'inference_results.json')
        logger.info('Results saved to {}', json_path)

    # Print summary
    total_alleles = result.total_alleles
    total_markers = result.total_markers_called
    logger.info(
        'Inference complete: {} profiles, {} markers called, {} total alleles',
        result.total_profiles,
        total_markers,
        total_alleles,
    )
