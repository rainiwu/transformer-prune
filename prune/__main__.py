import prune.models as pm
import prune.train as pr

from prune.pruners import neural

if __name__ == "__main__":
    pr.finetune_squad(
        *pm.PRETRAINED["squad"]["gpt2"],
        prune_generator=neural.generate_pruning_functions
    )
