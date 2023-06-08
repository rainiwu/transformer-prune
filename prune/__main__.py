import copy
import functools

import numpy as np
import torch

import prune.models as pm
import prune.train as pr
from prune.pruners import neural, platon

if __name__ == "__main__":
    for model in ["gpt2", "bert", "albert", "roberta", "electra"]:
        # generate initial results without pruning
        print("training ", model)
        tuned = pr.finetune_squad(
            *pm.PRETRAINED["squad"][model], num_epochs=1, dataset_percent=10  # type: ignore
        )
        torch.save(tuned.state_dict(), model + "_default_10percent_1epoch.pth")

        # apply various levels of finetuned pruning
        for value in np.linspace(0.05, 1, num=10):
            platon_config = copy.deepcopy(platon.DEFAULT_CONFIG)
            platon_config["final_threshold"] = value
            platon_pruning_function = functools.partial(
                platon.generate_pruning_functions, default=platon_config
            )
            platon_pruned = pr.finetune_squad(
                *pm.PRETRAINED["squad"][model],  # type: ignore
                num_epochs=1,
                dataset_percent=10,
                prune_generator=platon_pruning_function,
            )
            torch.save(
                platon_pruned.state_dict(),
                model + "_platon_" + f"{value}" + "_10percent_1epoch.pth",
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
