"""
performs pruning using built-in torch methods
"""
from typing import List, Tuple

import torch
from torch.nn.utils import prune


def random_prune(model: torch.nn.Module, sparsity: float) -> None:
    parameters: List[Tuple[torch.nn.Module, str]] = []
    for name, module in model.named_modules():
        parameters.append((module, name))
    prune.global_unstructured(
        parameters, pruning_method=prune.RandomUnstructured, amount=sparsity
    )


def lowest_prune(model: torch.nn.Module, sparsity: float) -> None:
    parameters: List[Tuple[torch.nn.Module, str]] = []
    for name, module in model.named_modules():
        parameters.append((module, name))
    prune.global_unstructured(
        parameters, pruning_method=prune.L1Unstructured, amount=sparsity
    )
