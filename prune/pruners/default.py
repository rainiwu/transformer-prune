"""
performs pruning using built-in torch methods
"""

import torch
from torch.nn.utils import prune


def random_prune(model: torch.nn.Module, sparsity: float) -> None:
    prune.global_unstructured(
        model.parameters(), pruning_method=prune.RandomUnstructured, amount=sparsity
    )


def lowest_prune(model: torch.nn.Module, sparsity: float) -> None:
    prune.global_unstructured(
        model.parameters(), pruning_method=prune.L1Unstructured, amount=sparsity
    )
