# BaseH3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction
The goal of this blog post is to propose a new model, so-called BaseH3D-Net, that builds on the paper “H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction [^2]”. We are doing this for an assignment of the course CS4245 Seminar Computer Vision by Deep Learning (2021/22 Q4) at Delft University of Technology.

Click [here](https://hackmd.io/7VpIn0GFTTCApcEB_wX31A?view) to view this blog post online. 

For those who are new to these topic, click [here](https://hackmd.io/@Group3-H3D-Net/BJqxCkmm9) to check our reproducibility project of H3D-Net in which we explains the methods in more detail.

## Introduction

### Original Paper
The “H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction” paper introduces a new high-fidelity full 3D head reconstruction method called H3D-Net that outperforms state-of-the-art models, such as MVFNet, DFNRMVS and IDR, in the few-shot (3 views) scenario. The H3D-Net utilizes both DeepSDF (a learned shape prior) and IDR[^1] (fine-tuning details) to achieve fast high-fidelity 3D face reconstruction from 2D images with different views. Please check the papers for more background information about DeepSDF and IDR. 

The image below shows the process used by H3D-Net. The training and inference process (3D Prior), which uses DeepSDF to learn a prior model, is shown on the left side. The reconstruction process is illustraited on the right side, that uses the learned prior as a starting point for the IDR implementation, to get the finer details of the 3D model.

![](https://i.imgur.com/aKPPEg3.jpg)  
*Figure 1: H3D-Net implementation*

### Our Approach

While trying to reproduce the original H3D-Net paper, we realised it was extremely difficult to implement the DeepSDF prior. That's when we came up with an idea to instead just use a single IDR prior, this would mean less training time for the prior model and hopefully a comparable result to the final H3D-Net implementation.  

We will pre-train an IDR model to a particular number of epochs to create a rough human head reference that is not too detailed or too simple and use that as prior when training on a new sample. This embarassingly simple prior has shown good results and on par performance compared to the original  H3D-Net method that uses IDR with a DeepSDF prior. Due to the complexity and heavy training required to make the DeepSDF prior, it's more than welcome to make this process easier. The simple prior we used is based on 500 epoch training for one scan. Our method is tested on H3DS dataset for 10 different scans which was also used in the results of the H3D-Net paper. Because in their paper it was already concluded that the use of the DeepSDF prior outperforms IDR from scratch for few-shots 3D Head Reconstruction, we will consider only the few-shot scenario (3 views).

## Methods
To use a pre-trained prior as input for IDR method for a new scan. We have to copy the .ply file (learnable paramters of all layers) to the folder of the new scan. 

Check [here](https://hackmd.io/@Group3-H3D-Net/BJqxCkmm9) our previous work to see how to set up cloud computing. 

Make new scan directory for the new sample and copy the  prior .ply file into it.

``` bash
# start up vm
sudo su
nvidia-smi -pm 1
# ctrl + D
conda activate idr

# Make new scan directory with Prior
# Fill in SCAN_ID 
cd H3D-Net-new/IDR/exps/
mkdir H3D_fixed_cameras_SCAN_ID/  
cp -a H3D_fixed_cameras_2/2022_05_18_12_53_17/ H3D_fixed_cameras_SCAN_ID/
```

We are now ready to train on differen scans based on 3 views. In order to train, we used the following code:

``` bash
cd ../code
true > nohup.out

# Training
# Fill in SCAN_ID 
nohup python training/exp_runner.py --conf ./confs/H3D_fixed_cameras_3.conf --scan_id SCAN_ID --is_continue --nepoch 2500 --timestamp 2022_05_18_12_53_17 --checkpoint 500 &

# check status
jobs -l
nano nohup.out
# Alt + / 
```

Once finished training a specific view on a specific scene, we need to generate the`surface_world_coordinates.ply`, run the following code in the terminal:

``` bash
# Evaluation
# Fill in SCAN_ID 
python evaluation/eval.py  --conf ./confs/H3D_fixed_cameras_3.conf --scan_id SCAN_ID
```

We trained all models with a non-decaying learning rate of $1.0e^{-4}$ and 2000 epochs ontop of the prior.

## Initial Experiments 
In our first experiment, we have chosen 3 different scans from the H3DS dataset that could be used as a pre-trained prior. Note that only the prior is trained on 32 views as it will allow for more details of the head reconstruction to be rendered and preserving the facial features of the ground truth. To check the effect of the number of epochs of pre-training a certain prior, we will examine the results for a new scan. For this experiment, we have used scan 1, 2 and 10 as prior and evaluated it on scan 3. Because scan 1, 2 and 10 are all men with not much hair, we used scan 3 which is woman to check wehther it generalizes well to a different gender. Figure 2 shows Scan 2 as the prior, and the differences associated with this prior based on how many epochs it was trained for. You can observe slight improvements in facial details of the prior as the number of epochs increases.  

![](https://i.imgur.com/zpTHkBu.png)
*Figure 2: Scan 2 Prior trained at different Epochs*

As can be seen in Table 1 below, the amount of epochs the prior is pre-trained for introduces a different relationship depending on the prior that is chosen. 
- For **scan 10**, it shows a trade-off between the average surface error in millimeters for the face and head depending on the number of epochs trained. Low face error but larger head error for the 500 epoch prior. Likewise, high face error but lower head error for the 2000 epoch prior. 
- For **scan 1**, it looks like that both head and face metric are independent on the number of epochs used.
-  For **scan 2**, it shows to have relatively low face and head error at 500 epochs. However, both the errors increase with the higher number of epochs pre-trained prior.

There seems to be no real structure that can be deduced by this ibservation, except that for the prior trained on 500 epochs, the facial error tended to be less, therefore we decided to train our prior for 500 epochs, as we were more interested in facial detail than the head.

This face/head metric was first introduced in the H3D-Net paper to compare the performance between H3D-Net and IDR methods. The lower the value the better the result.

*Table 1: **Pre-trained prior comparison**. Average surface error in millimeters computed for scan 3 based scan 1, 2 and 10 pre-trained prior. The details of the face/head metric can found in their paper [^1].*
<table>
   <tr>
      <th rowspan="2">Prior</th>
      <th colspan="2">500 epochs</th>
      <th colspan="2">1000 epochs</th>
      <th colspan="2">2000 epochs</th>
   </tr>
   <tr>
      <th>face</th>
      <th>head</th>
      <th>face</th>
      <th>head</th>
      <th>face</th>
      <th>head</th>
   </tr>
   <tr>
      <td>Scan10</td>
      <td>1.46</td>
      <td>9.92</td>
      <td>1.87</td>
      <td>7.98</td>
      <td>2.09</td>
      <td>7.49</td>
   </tr>
   <tr>
      <td>Scan1</td>
      <td>1.91</td>
      <td>9.96</td>
      <td>1.92</td>
      <td>9.94</td>
      <td>1.91</td>
      <td>8.68</td>
   </tr>
   <tr>
      <td>Scan2</td>
      <td>1.26</td>
      <td>7.78</td>
      <td>2.21</td>
      <td>9.06</td>
      <td>2.20</td>
      <td>10.1</td>
   </tr>
    <tr>
      <td>AVG</td>
      <td>1.54 &#177 0.27</td>
      <td>9.22 &#177 1.01</td>
      <td>2.00 &#177 0.15 </td>
      <td>8.99 &#177 0.80</td>
      <td>2.07 &#177 0.12</td>
      <td>8.76 &#177 1.07</td>
   </tr>
</table>

We will continue the final evaluation using the **scan 2 prior pre-trained on 500 epochs** because of the low error for both the face and head metric. We think that the lower the epoch pre-training, the more general head reconstruction is rendered since it will have less details to start with. Higher epoch pre-trained models already have too fine-detailed features as it has all the facial features for that particular scan only and probably doesn't generalizes well to new scans. The prior's features translates into a new scan and likely causing a larger error since the evaluation method sees these facial features as flaws eventough it look more smooth and better to the human eye, see figure 3. However, these findings should be investigated more deeply by checking whether the metric actually makes fair comparisons and is not misleading.



![](https://i.imgur.com/067yl9S.png)
*Figure 3: Scan 3 Evaluation with Scan 2 Prior trained at different Epochs*





## Results
In this section, we disuss the results of training IDR on the H3DS dataset for 10 different scans using our simple prior of scan 2 trained for 500 epochs and only using 3 views. In the paper of H3D-Net, only the average of the 10 different scans was shown, therefore we also show only the average result on all 10 scans. Table 2 below shows these results. To see the results per scan, please look at the images provided in the Appendix.

We can see that our method performs on par to H3D-Net for the face metric and outperforms H3D-Net for the head metric. However, we can see a lot of variance for the head metric, see figure 4. 


In the file `data_results.py` you can find all individual results for each scan. Because the results of H3D-Net didn't only show the average of the 10 evaluated scans, it is hard to make any conclusion based on our results. 

*Table 2: **Few-Shot (3 views) 3D Head Reconstruction comparison**. Average surface error in millimeters computed over all 10 subjects in the H3DS dataset.*
| Method               | face             | head            |
| -------------------- | ---------------- | --------------- |
| IDR [^1]             | 3.52             | 17.04           |
| H3D-Net [^2]         | 1.49             | 12.76           |
| BaseH3D-Net (ours)   | **1.48** ± 0.30  | **10.65** ± 3.42|

<img src="https://i.imgur.com/rY6IJtS.png"  width=50% height=50%>
<img src="https://i.imgur.com/X0FI8EY.png"  width=50% height=50%>
![](https://i.imgur.com/X0FI8EY.png)

*Figure 4: **BaseH3D-Net** results of the evaluated 10 subjects in the H3DS dataset*

Figure 5 and Fifure 6 below show the drastic improvement that our method can have over a simple IDR method. In order to get these results, the "Only IDR" models were trained from scratch using 3 views for 2000 epochs. Our BaseH3D method uses scan 2 trained at 500 epochs as the prior and then each scan is trained for a further 2000 epochs. As you can see the results are quite an improvement. Please see the Appendix for more scan comparisons.

![](https://i.imgur.com/HdqkCvZ.png)  
*Figure 5: Scan 6 Results*

![](https://i.imgur.com/c2zlyZI.png)  
*Figure 6: Scan 9 Results*

## Conclusion

To conclude this blog post, we would like to state that our method by no means can replace the method of H3D-Net. This is because our method largely depends on the prior model selected. In our case we selected a prior which managed to generalize well to the H3D Dataset, however it may perform worse on other datasets with different facial features. 

While we cannot replace the H3D-Net method with ours, we are excited that we managed to get a result almost on par with theirs. This definitely proves that using any prior would result in better performance.

We would recommend using our method if you are low on computational resources, or looking for a faster training method than using IDR alone. We would also reccomend to try at least 3 different prior scans and select the best performing one, as the results can vary drastically per prior.


## Future Work
We have tried to create a more general prior by averaging the learnable parameters of multiple scans (e.g. scan 1, 2 and 10), see `avg_model_parameters.py`. We think that this might even show better performance. Eventhough we only made changes to the values of the .ply file, it gave an error, see figure 7, when training on this average prior which we were unable to solve. 

![](https://i.imgur.com/LIiNgvJ.jpg) \
*Figure 7: Error during training of an averaged prior*

To support our conclusion that this embarassing simple prior can perform on par to the original H3D-Net method on few-shot 3D head reconstruction, we should evaluate this with more scans and more comparable date. However, this dataset contains only 22 different scans, and since we only evaluated one prior model, we cannot draw any concrete conclusions. We would like to encourage others to continue challenging complicated papers with simpler methods.  

## Contributions

* Alon 
    - IDR Training and Evaluation - Scans: 1, 2, 4, 7, 9 and 10
    - Reproduced model Landmarks: *reproduce.ipynb*, *All scanID landmarks.txt files using FreeCAD*
    - Final ID Evalualtion and Results Processing
    - Contribute to the blogpost (the images of the 3D head reconstruction)
    - Conclusion 
* Max 
    - IDR Training and Evaluation - Scans: 6, 3, 5 and 8
    - Tried avaraging the model parameters
    - Anlysis of the results
    - Wrote the blogpost 




## Appendix

### Final Results Comparison

![](https://i.imgur.com/HdqkCvZ.png)
-----------------------------------

![](https://i.imgur.com/1VChTYT.png)
-----------------------------------

![](https://i.imgur.com/c2zlyZI.png)
-----------------------------------

![](https://i.imgur.com/oDoZfTj.png)
-----------------------------------

![](https://i.imgur.com/TUYQL4l.png)
-----------------------------------

![](https://i.imgur.com/UPUaTWJ.png)
-----------------------------------

![](https://i.imgur.com/VgqMI4r.png)
-----------------------------------

![](https://i.imgur.com/zNcKgUJ.png)
-----------------------------------

![](https://i.imgur.com/IpqDBzn.png)
-----------------------------------

### Computational Results

![](https://i.imgur.com/lOFUD1B.png)

![](https://i.imgur.com/9WaK5lt.png)

![](https://i.imgur.com/AwffCGt.png)

![](https://i.imgur.com/mntsk0r.png)

![](https://i.imgur.com/uTfgSFR.png)

![](https://i.imgur.com/0ebN8X2.png)

![](https://i.imgur.com/hj7POxU.png)

![](https://i.imgur.com/cdtPrcd.png)

![](https://i.imgur.com/FmPlBM8.png)

![](https://i.imgur.com/6jR9Cu3.png)
 

## References

[^1]: Yariv, L., Kasten, Y., Moran, D., Galun, M., Atzmon, M., Ronen, B., & Lipman, Y. (2020). Multiview neural surface reconstruction by disentangling geometry and appearance. Advances in Neural Information Processing Systems, 33, 2492-2502. https://doi.org/10.48550/arXiv.2003.09852


[^2]: Ramon, E., Triginer, G., Escur, J., Pumarola, A., Garcia, J., Giro-i-Nieto, X., & Moreno-Noguer, F. (2021). H3d-net: Few-shot high-fidelity 3d head reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 5620-5629). https://doi.org/10.48550/arXiv.2107.12512