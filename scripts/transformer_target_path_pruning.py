import copy
import prune.models as pm
import torch
import prune.evaluate as eval
# pruning imports
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import csv
import pandas as pd

# only tested on huggingface gpt2
def target_path_pruning(model, target_path, prune_rate):
    assert isinstance(target_path,str)
    assert isinstance(prune_rate,float)
    assert prune_rate > 0 or prune_rate < 1
    assert target_path == "queries" or target_path == "values" or target_path == "keys"

    hidden_size = model.config.hidden_size
    prune_count_test = 0
    prune_count = 0
    for name, module in model.named_modules():

        # Identifying the projection matrices that generate queries, keys, and values
        if name.endswith("c_attn"):

            # Performing Pruning on the projection matrix
            prune.l1_unstructured(module, name='weight', amount=prune_rate)
            # Perfomring Pruning on the bias vector
            prune.l1_unstructured(module, name='bias', amount=prune_rate)
            
            # pulling out the binary mask
            weight_mask = module.get_buffer("weight_mask")
            bias_mask = module.get_buffer("bias_mask")

            # ======= Novel Idea Alert! ======= #
            if target_path == "queries":
                # preserving all the non-query elements to 1
                weight_mask[:,hidden_size:] = 1
                bias_mask[hidden_size:] = 1
            elif target_path == "keys":
                # perserving all the non-key elements
                weight_mask[:,:hidden_size] = 1
                weight_mask[:,2*hidden_size:] = 1
                bias_mask[:hidden_size] = 1
                bias_mask[2*hidden_size:] = 1

            elif target_path == "values":
                # preserving all the non-value elements to 1
                weight_mask[:,:2*hidden_size] = 1
                bias_mask[:2*hidden_size] = 1

            # counting the number of zeros
            prune_count += torch.numel(weight_mask) - torch.count_nonzero(weight_mask)
            prune_count += torch.numel(bias_mask) - torch.count_nonzero(bias_mask)

            # adding back in the modified binary mask
            module.register_buffer("weight_mask",weight_mask)
            module.register_buffer("bias_mask",bias_mask)


            # applying the pruning operation
            prune.remove(module, 'weight')
            prune.remove(module, 'bias')
    
    return model, prune_count.item()


'''

# Target path pruning experiment:


token, base = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])

_, model_pruned_queries = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])
_, model_pruned_keys = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])
_, model_pruned_values = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])

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
'''