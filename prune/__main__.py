import copy
import functools

import numpy as np
import torch

import prune.models as pm
import prune.train as pr
from prune.pruners import neural, default

if __name__ == "__main__":
    for model in ["gpt2", "bert", "albert", "roberta", "electra"]:
        # generate initial results without pruning
        print("training ", model)
        tuned = pr.finetune_squad(
            *pm.PRETRAINED["squad"][model], num_epochs=1, dataset_percent=10  # type: ignore
        )
        torch.save(tuned.state_dict(), model + "_default_10percent_1epoch.pth")

        # apply various levels of random, l1 pruning, and finetuned pruning
        for value in np.linspace(0, 1, num=10):
            random_pruned = copy.deepcopy(tuned)
            default.random_prune(random_pruned, sparsity=value)
            torch.save(
                random_pruned.state_dict(),
                model + "_random_pruned_" + f"{value}" + "_10percent_1epoch.pth",
            )

            l1_pruned = copy.deepcopy(tuned)
            default.lowest_prune(l1_pruned, sparsity=value)
            torch.save(
                l1_pruned.state_dict(),
                model + "_l1_pruned_" + f"{value}" + "_10percent_1epoch.pth",
            )

            nc_config = copy.deepcopy(neural.DEFAULT_CONFIG)
            nc_config[0]["target_sparsity"] = value
            pruning_function = functools.partial(
                neural.generate_pruning_functions, configs=nc_config
            )
            nc_pruned = pr.finetune_squad(
                *pm.PRETRAINED["squad"][model],  # type: ignore
                num_epochs=1,
                dataset_percent=10,
                prune_generator=pruning_function,
            )
            torch.save(
                nc_pruned.state_dict(),
                model + "_nc_" + f"{value}" + "_10percent_1epoch.pth",
            )
