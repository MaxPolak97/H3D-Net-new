import torch
from collections import Counter
from collections import OrderedDict

# Load Model Parameters of different scans
model = torch.load('baseline_scan10/checkpoints/ModelParameters/1000.pth', map_location='cpu')
model2 = torch.load('baseline_scan10/checkpoints/ModelParameters/1200.pth', map_location='cpu')

print(type(model))  # <class 'dict'>
print(model.keys())  # dict_keys(['epoch', 'model_state_dict'])

print(type(model['epoch']))  # <class 'int'>
print(model['epoch'])  # 1000
print(type(model['model_state_dict']))  # <class 'collections.OrderedDict'>

model_state_dict = model['model_state_dict']
print(model_state_dict.keys())  # odict_keys(['implicit_network.lin0.bias', 'implicit_network.lin0.weight_g', etc.
# print(model_state_dict['implicit_network.lin0.bias'])
#print(model_state_dict['implicit_network.lin0.weight_v'])

model_state_dict2 = model2['model_state_dict']

sum_model = model_state_dict['implicit_network.lin0.bias'] + model_state_dict2['implicit_network.lin0.bias']
avg_model = sum_model / 2

print(avg_model[0], model_state_dict['implicit_network.lin0.bias'][0], model_state_dict2['implicit_network.lin0.bias'][0])


