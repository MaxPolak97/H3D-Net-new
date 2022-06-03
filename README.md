# BaseH3D-Net
The goal of this blog post is to propose a new model, so-called BaseH3D-Net, that builds on the paper “H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction [^1]”. We are doing this for an assignment of the course CS4245 Seminar Computer Vision by Deep Learning (2021/22 Q4) at Delft University of Technology.

Click [here](https://hackmd.io/7VpIn0GFTTCApcEB_wX31A?view) to view this blog post online. 

We will pre-train an IDR model to a particular number of epochs to create a rough human head reference that is not too detailed or too simple and use that as prior when training on a new sample. This embarassing simple prior has  similar and sometimes slightly better performance than the H3D-Net method that uses IDR with a DeepSDF prior. The simple prior we used is based on 500 epoch training. Our method is tested on H3DS dataset for 10 different scans which was also used in the results of the H3D-Net paper. Because in their paper it was already concluded that the use of the DeepSDF prior outperforms IDR from scratch for few-shots 3D Head Reconstruction, we will consider only the few-shot scenario (3 views).


## Experiments 
In our first experiment, we have chosen 3 different scans from the H3DS dataset that could be used as a pre-trained prior. To check the effect of the number of epochs of pre-training a certain the prior, we will examine the results for a new scan. For this experiment, we have used scan 1, 2  and 10 as prior and evaluated it on scan 3. Because scan 1, 2 and 10 are both man with not much hair, we used scan 3 which is woman to check wehther it generalizes well to a different gender. Below you can see the figure of all 4 scans. 

As can be seen in the table below the amount of epochs the prior is pre-trained, introduces a trade-off between the average surface error in millimeters for the face and head for both scan 10 and scan 2. This face/head metric was first introduced in the H3D-Net paper to compare the performance between H3D-Net and IDR methods. However, scan 1 prior has a different behavior and doesn't do well for the face metric  which we think that has to do with the weird features, see figure 2. 

*Table 2: **3D head reconstruction method comparison**. Average surface error in millimeters computed over all 10 subjects in the H3DS dataset. The details of the face/head metric can found in their paper [^1].*
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
      <td></td>
      <td>1.54 &#177 0.27</td>
      <td>9.22 &#177 1.01</td>
      <td>2.00 &#177 0.15 </td>
      <td>8.99 &#177 0.80</td>
      <td>2.07 &#177 0.12</td>
      <td>8.76 &#177 1.07</td>
   </tr>
</table>

Because scan 10 was used in their paper for evaluating the model performance and scan 1 shows relatively worse performance, we will continue the final evaluation using scan 2 with 500 epochs as prior because of the low error for the face. We choose 500 epochs since it balances both the face and head region. In addition, the lower the epoch pr-training, the more general head reconstruction is rendered as it has less details. 

## Results
In this section, we disuss the results of training IDR on the H3DS dataset for 10 different scans using our simple prior. In the paper of H3D-Net, only the average of the 10 different scans was shown. We can see that our method performs on par to H3D-Net for the face metric and outperforms H3D-Net for the head metric. However, we can see a lot of variance for the head metric. In the file ``data_results`` you can find all individual results for each scan. 

| Method               | face             | head            |
| -------------------- | ---------------- | --------------- |
| IDR [^1]             | 3.52             | 17.04           |
| H3D-Net [^2]         | 1.49             | 12.76           |
| BaseH3D-Net (ours)   | **1.48** ± 0.30  | **10.65** ± 3.42|







## References
[^1]: Yariv, L., Kasten, Y., Moran, D., Galun, M., Atzmon, M., Ronen, B., & Lipman, Y. (2020). Multiview neural surface reconstruction by disentangling geometry and appearance. Advances in Neural Information Processing Systems, 33, 2492-2502. https://doi.org/10.48550/arXiv.2003.09852


[^2]: Ramon, E., Triginer, G., Escur, J., Pumarola, A., Garcia, J., Giro-i-Nieto, X., & Moreno-Noguer, F. (2021). H3d-net: Few-shot high-fidelity 3d head reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 5620-5629). https://doi.org/10.48550/arXiv.2107.12512




