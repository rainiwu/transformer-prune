from typing import Callable, Dict

import torch

import prune.train as pr


def evaluate_squad_model(
    tokenizer: Callable, model: torch.nn.Module, quantize: bool = False
) -> Dict:
    _, eval_dataloader, validation_dataset = pr.generate_squad_dataloaders(
        tokenizer, 100
    )
    if quantize is True:
        pr.evaluate_model(model, validation_dataset, eval_dataloader, quantize=True)
    return pr.evaluate_model(model, validation_dataset, eval_dataloader)


def get_sparsity(model: torch.nn.Module) -> float:
    total_params: int = 0
    num_nonzero: int = 0
    for _, param in model.named_parameters():
        total_params += torch.numel(param)
        num_nonzero += int(torch.count_nonzero(param))
    return 1 - (num_nonzero) / total_params


if __name__ == "__main__":
    pass
