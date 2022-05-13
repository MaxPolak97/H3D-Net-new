import torch
from collections import Counter
from collections import OrderedDict

# Load Model Parameters of different scans
model = torch.load('model_baseline_scans/baseline_scan10/checkpoints/ModelParameters/2000.pth')


print(type(model))  # <class 'dict'>
print(model.keys())  # dict_keys(['epoch', 'model_state_dict'])

print(type(model['epoch']))  # <class 'int'>
print(model['epoch'])  # 1000
print(type(model['model_state_dict']))  # <class 'collections.OrderedDict'>

model_state_dict = model['model_state_dict']
print(model_state_dict.keys())  # odict_keys(['implicit_network.lin0.bias', 'implicit_network.lin0.weight_g', etc.
# print(model_state_dict['implicit_network.lin0.bias'])
#print(model_state_dict['implicit_network.lin0.weight_v'])


