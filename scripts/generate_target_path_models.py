import copy
import prune.models as pm
import torch
import prune.evaluate as eval
# pruning imports
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import csv
import pandas as pd
from transformer_target_path_pruning import target_path_pruning


prune_rate_dict = {
    "queries" : [],
    "keys"    : [],
    "values"  : [],
    "nc"      : []
}

paths = ["queries","keys","values"]

prune_rates = [.1,.2,.3,.4,.5,.6,.7,.8,.9]

gpt2_squad_model_path = "gpt2_default_100percent_5epoch.pth"
gpt2_size = 124000000
for path in paths:
    for prune_rate in prune_rates:
        # initializing model
        _, model = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])
        # loading in squad weights
        model.load_state_dict(torch.load(gpt2_squad_model_path))
        # performing target-path pruning
        model, prune_count = target_path_pruning(model,path,prune_rate)
        # calculating actual prune rate
        actual_prune_rate = prune_count/gpt2_size
        # adding prune rate to dictionary
        prune_rate_dict[path].append(actual_prune_rate)
        # saving pruned model
        model_str = "gpt_target_path_pruning_"+ path+"_"+ str(prune_rate) +".pt"
        torch.save(model.state_dict(), model_str)

prune_rate_df = pd.DataFrame.from_dict(prune_rate_dict)
prune_rate_df['nc'] = prune_rate_df.mean(axis=1)
prune_rate_df.to_csv("Actual_Prune_Rates.csv",index=False)

