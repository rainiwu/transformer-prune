"""
setup neural_compressor pruning boilerplate
"""

from typing import Dict, Union, List

from neural_compressor.training import prepare_compression, WeightPruningConfig
import torch

from prune.types import TrainerHook

DEFAULT_CONFIG: List[Dict[str, Union[int, float, str]]] = [
    {
        "start_step": 1,  # Step at which to begin pruning, if a gradient-based criterion is used (e.g., snip-momentum), start_step should be equal to or greater than 1.
        "end_step": 10000,  # Step at which to end pruning, for one-shot pruning start_step = end_step.
        "target_sparsity": 0.9,  # Target sparsity ratio of modules.
        "pruning_frequency": 250,  # Frequency of applying pruning, The recommended setting is one fortieth of the pruning steps.
        "pattern": "4x1",  # Default pruning pattern.
    }
]


def generate_pruning_functions(model: torch.nn.Module) -> TrainerHook:
    config = WeightPruningConfig(DEFAULT_CONFIG)
    compression_manager = prepare_compression(model, config)
    return TrainerHook(
        pretrain=compression_manager.callbacks.on_train_begin,
        prestep=compression_manager.callbacks.on_step_begin,
        preoptimize=compression_manager.callbacks.on_before_optimizer_step,
        postoptimize=compression_manager.callbacks.on_after_optimizer_step,
        posttrain=compression_manager.callbacks.on_train_end,
    )
