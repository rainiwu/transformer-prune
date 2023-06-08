from typing import Callable, Dict

import torch

import prune.train as pr


def evaluate_squad_model(tokenizer: Callable, model: torch.nn.Module) -> Dict:
    _, eval_dataloader, validation_dataset = pr.generate_squad_dataloaders(
        tokenizer, 100
    )
    return pr.evaluate_model(model, validation_dataset, eval_dataloader)
