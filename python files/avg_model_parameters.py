import torch
from collections import OrderedDict

# Load Model Parameters of different scans
model1 = torch.load('model_baseline_scans/baseline_scan10/checkpoints/ModelParameters/500.pth', map_location='cpu')
model2 = torch.load('model_baseline_scans/baseline_scan6/checkpoints/ModelParameters/500.pth', map_location='cpu')
model3 = torch.load('model_baseline_scans/baseline_scan22/checkpoints/ModelParameters/500.pth', map_location='cpu')

model_state_dict1 = model1['model_state_dict']
model_state_dict2 = model2['model_state_dict']
model_state_dict3 = model3['model_state_dict']

avg_model_state_dict = OrderedDict() # create new state dictionary with average model parameters
for k in model_state_dict1.keys():
    sum_model = model_state_dict1[k] + model_state_dict2[k] + model_state_dict3[k]
    avg_model_state_dict[k] = sum_model / 3

print(avg_model_state_dict.keys())

avg_model = {'epoch': model1['epoch'], 'model_state_dict': avg_model_state_dict}
print(avg_model.keys()) # dict_keys(['epoch', 'model_state_dict'])

torch.save(avg_model, 'prior_500.pth', _use_new_zipfile_serialization=False)

