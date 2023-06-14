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


# Target path pruning experiment:
gpt2_squad_model_path = "gpt2_default_100percent_5epoch.pth"

token, base = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])

_, model_pruned_queries = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])
_, model_pruned_keys = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])
_, model_pruned_values = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])

base.load_state_dict(torch.load(gpt2_squad_model_path))
model_pruned_queries.load_state_dict(torch.load(gpt2_squad_model_path))
model_pruned_keys.load_state_dict(torch.load(gpt2_squad_model_path))
model_pruned_values.load_state_dict(torch.load(gpt2_squad_model_path))

# building separate models that prune queries, keys, and values
model_pruned_queries, query_prune_count = target_path_pruning(model_pruned_queries, "queries", 0.5)

model_pruned_keys, key_prune_count = target_path_pruning(model_pruned_keys, "keys", 0.5)

model_pruned_values, value_prune_count = target_path_pruning(model_pruned_values, "values", 0.5)

# getting a count of how many elements of gpt2 were pruned
print("Query Prune Count: ", query_prune_count)
print("Key Prune Count: ", key_prune_count)
print("Value Prune Count: ", value_prune_count)

# evaluating on squad test set
baseline_results = eval.evaluate_squad_model(token, base)

query_pruning_results = eval.evaluate_squad_model(token, model_pruned_queries)

key_pruning_results = eval.evaluate_squad_model(token, model_pruned_keys)

value_pruning_results = eval.evaluate_squad_model(token, model_pruned_values)

print("Base Results: ", baseline_results)
print("Query Pruning Results: ", query_pruning_results)
print("Key Pruning Results: ", key_pruning_results)
print("Value Pruning Results: ", value_pruning_results)

# saving the state dict of these models:
def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = [value , dict_1[key]]
   return dict_3

final_results_dict = mergeDictionary(baseline_results, query_pruning_results)
final_results_dict = mergeDictionary(final_results_dict,key_pruning_results)
final_results_dict = mergeDictionary(final_results_dict,value_pruning_results)
final_results_df = pd.DataFrame.from_dict(final_results_dict)

final_results_df.to_csv("target_path_pruning_initial_results.csv",index=False)