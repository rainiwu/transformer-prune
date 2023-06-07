import prune.models as pm
import prune.train as pr

pr.finetune_squad(*pm.PRETRAINED["squad"]["gpt2"])
