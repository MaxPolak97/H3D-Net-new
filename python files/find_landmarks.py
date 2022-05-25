from h3ds.dataset import H3DS
#import numpy as np
#from tempfile import TemporaryFile
#import trimesh
#import copy
#from ipywidgets import interact, interactive, widgets, fixed
from h3ds.mesh import Mesh
#from vtkplotter import *

h3ds = H3DS(path='h3ds')

scene_id = '3b5a2eb92a501d54'


mesh_pred_path = 'idr_eval_results/reconstructions/idr/' + scene_id + '_3.ply'
mesh_gt_path = 'h3ds_v0.2/' + scene_id + '/full_head.obj'

mesh_pred = Mesh().load(mesh_pred_path)
mesh_gt, images, masks, cameras = h3ds.load_scene('3b5a2eb92a501d54', '3')

vertices = mesh_gt.vertices
vertices_pred = mesh_pred.vertices

closest_vert_right_eye = vertices_pred[0]
closest_vert_left_eye = vertices_pred[0]
closest_vert_nose_base = vertices_pred[0]
closest_vert_right_lips = vertices_pred[0]
closest_vert_left_lips = vertices_pred[0]
closest_vert_nose_tip = vertices_pred[0]

closest_vert = [closest_vert_right_eye,
                closest_vert_left_eye,
                closest_vert_nose_base,
                closest_vert_right_lips,
                closest_vert_left_lips,
                closest_vert_nose_tip]

#print(closest_vert)
#168f8ca5c2dce5bc
#right_eye = [-64.928, -103.996, 552.295]
#left_eye = [-37.282, -105.554, 563.712]
#nose_base = [-47.800, -34.497, 542.937]
#right_lips = [-84.489, 6.352, 558.372]
#left_lips = [-13.259, 8.071, 561.263]
#nose_tip = [-48.228, -65.305, 560.791]

#3b5a2eb92a501d54
right_eye = [-10.862, -48.422, -103.338]
left_eye = [14.263, -52.190, -103.028]
nose_base = [-0.184, -11.363, -118.767]
right_lips = [-24.052, 20.042, -99.844]
left_lips = [22.428, 18.786, -100.336]
nose_tip = [1.700, -32.091, -132.616]

landmarks = [right_eye, left_eye, nose_base, right_lips, left_lips, nose_tip]


print(landmarks)
print(closest_vert[0])

closest_loss = []

for idx, vertex in enumerate(closest_vert):
    closest_loss.append(
        abs(vertex[0] - landmarks[idx][0]) + abs(vertex[1] - landmarks[idx][1]) + abs(vertex[2] - landmarks[idx][2]))




landmarks_idx = [0, 0, 0, 0, 0, 0]

for idx_1, landmark in enumerate(landmarks):
    for idx, vert in enumerate(vertices_pred):
        loss = abs(vert[0] - landmark[0]) + abs(vert[1] - landmark[1]) + abs(vert[2] - landmark[2])
        if loss < closest_loss[idx_1]:
            closest_vert[idx_1] = vert
            closest_loss[idx_1] = loss
            landmarks_idx[idx_1] = idx

print(closest_loss)

print(landmarks_idx)