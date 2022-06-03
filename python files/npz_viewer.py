# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:04:01 2022

@author: Maxpo
"""

import numpy as np
from h3ds.dataset import H3DS
h3ds_path = 'h3ds_v0.2'
h3ds = H3DS(path=h3ds_path)
import npzviewer as npzview



# IDR format
#data = np.load('IDR/data/DTU/scan65/cameras.npz')
#data = np.load('IDR/data/H3D/view3/scan6/cameras.npz')
#data = np.load('h3')

#for keys in data.files:
    #print(keys)
    
#print(data['world_mat_0'])
#print(data['camera_mat_inv_0'])
#print(data['scale_mat_0'])

# H3D-Net format
#data1 = np.load('h3ds_v0.2/3b5a2eb92a501d54/cameras.npz')


#print(data1.files)
#for keys in data1.files:
    #print(keys)

#data2 = np.load('OWN_DATA/view3/scan6/cameras.npz')
#print(data2.files)
#for keys in data2.files:
    #print(keys)
#print(data['world_mat_0'])
#print(data['camera_mat_inv_0'])
#print(data['scale_mat_0'])

mesh, images, masks, cameras, cameras_OWN = h3ds.load_scene(scene_id='3b5a2eb92a501d54', views_config_id='3', normalized=True)

#print("Original Cameras file: ", cameras)
#print("Reproduced Cameras file: ", cameras_OWN)

loaded_cameras = np.load('h3ds_v0.2/3b5a2eb92a501d54/cameras.npz')
loaded_cameras_OWN = np.load('OWN_DATA/view3/scan6/cameras.npz')

scale_mats_original = [loaded_cameras['scale_mat_%d' % idx].astype(np.float32) for idx in range(3)]
world_mats_original = [loaded_cameras['world_mat_%d' % idx].astype(np.float32) for idx in range(3)]

scale_mats_OWN = [loaded_cameras_OWN['scale_mat_%d' % idx].astype(np.float32) for idx in range(3)]
world_mats_OWN = [loaded_cameras_OWN['world_mat_%d' % idx].astype(np.float32) for idx in range(3)]




print("Original Scale Mat:", scale_mats_original)
print("Own Scale Mat:", scale_mats_OWN)

print("Original World Mat:", world_mats_original)
print("Own World Mat:", world_mats_OWN)


