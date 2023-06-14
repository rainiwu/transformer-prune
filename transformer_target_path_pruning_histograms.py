from transformer_target_path_pruning import target_path_pruning
import copy
import prune.models as pm
import torch
import matplotlib.pyplot as plt
import numpy as np

token, base = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])

_, model_pruned_queries = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])
_, model_pruned_keys = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])
_, model_pruned_values = copy.deepcopy(pm.PRETRAINED["squad"]["gpt2"])

gpt2_squad_model_path = "gpt2_default_100percent_5epoch.pth"
base.load_state_dict(torch.load(gpt2_squad_model_path))
model_pruned_queries.load_state_dict(torch.load(gpt2_squad_model_path))
model_pruned_keys.load_state_dict(torch.load(gpt2_squad_model_path))
model_pruned_values.load_state_dict(torch.load(gpt2_squad_model_path))



# building separate models that prune queries, keys, and values
model_pruned_queries, query_prune_count = target_path_pruning(model_pruned_queries, "queries", 0.5)

model_pruned_keys, key_prune_count = target_path_pruning(model_pruned_keys, "keys", 0.5)

model_pruned_values, value_prune_count = target_path_pruning(model_pruned_values, "values", 0.5)

def weight_histogram(tensor, title, log=False):
    base_fc_layer = torch.flatten(tensor)
    base_fc_layer = base_fc_layer.detach().numpy()
    counts, bins = np.histogram(base_fc_layer, bins = 2048)
    zero_approximation = max(counts)
    element_count = len(base_fc_layer)
    print(type(counts))
    plt.hist(bins[:-1], bins, weights=counts, log=log)
    plt.title(title, fontsize=10)
    plt.xlabel("Weight Values")
    plt.xlim([-2.3, 2.3])
    if log:
        plt.ylabel("Count (Log Scale)")
    else:
        plt.ylabel("Count")
    

def get_relevant_model_tensors(model):
    relevant_tensors = [1, 6, 12]
    tensors = []
    count = 0
    for name, module in model.named_modules():
        # Identifying the projection matrices that generate queries, keys, and values
        if name.endswith("c_attn"):
             count += 1
             if count in set(relevant_tensors):
                weight_mask = module.get_parameter("weight")
                tensors.append(weight_mask)
    return tensors

base_tensors = get_relevant_model_tensors(base)
del base
query_pruned_tensors = get_relevant_model_tensors(model_pruned_queries)
del model_pruned_queries
key_pruned_tensors = get_relevant_model_tensors(model_pruned_keys)
del model_pruned_keys
value_pruned_tensors = get_relevant_model_tensors(model_pruned_values)
del model_pruned_values

relevant_tensors = [1, 6, 12]
ndx = 0
count = 1
for tensor in base_tensors:
    plt.subplot(4,3, count)
    weight_histogram(tensor,"Base Model, Layer: "+str(relevant_tensors[ndx]), log=True)
    ndx += 1
    count += 1
ndx = 0
for tensor in query_pruned_tensors:
    plt.subplot(4,3, count)
    weight_histogram(tensor,"Query Pruned Model, Layer: "+str(relevant_tensors[ndx]), log=True)
    ndx += 1
    count += 1
ndx = 0
for tensor in key_pruned_tensors:
    plt.subplot(4,3, count)
    weight_histogram(tensor,"Key Pruned Model, Layer: "+str(relevant_tensors[ndx]), log=True)
    ndx += 1
    count += 1
ndx = 0
for tensor in value_pruned_tensors:
    plt.subplot(4,3, count)
    weight_histogram(tensor,"Value Pruned Model, Layer: "+str(relevant_tensors[ndx]), log=True)
    ndx += 1
    count += 1
plt.suptitle("50% L1 Unstructured Targeted Pruning Across Projection Layers")
plt.tight_layout()
plt.show()