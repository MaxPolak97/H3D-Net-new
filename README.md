# BaseH3D-Net
The goal of this blog post is to propose a new model, so-called BaseH3D-Net, that builds on the paper “H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction”. We are doing this for an assignment of the course CS4245 Seminar Computer Vision by Deep Learning (2021/22 Q4) at Delft University of Technology.

We will pre-train an IDR model to a certain number of epochs to create a rough human head reference that is not too detailed or too simple and use that as prior when fitting on a new sample. We hope to get similar or even better results than the H3D-Net method that uses IDR with a DeepSDF prior for all different shots of views (e.g. 3, 4, 8, 16 and 32) and thus reducing training time by not starting from scratch (i.e. simple sphere) as the standalone IDR method used.


