Lp-spherical Sparse Training
=======================================

Controlling Sparsity of Neural Network with Weight Constrained on Unit Lp-Sphere.  
For more information, please contact author by: williamli_pro@163.com  

The corresponding paper is under review of Nature Machine Intelligence. 

# System Requirements
Operating systems: Windows or Linux  
Dependencies: CUDA, CuDNN, Python (version >= 3.5) Pytorch (version >= 1.2) and corresponding torchvision  

# Demo
After installed all of the requirements, please run demo.py  

# Results
* Procedures of LpSS  

![Procedures of LpSS](https://github.com/WilliamLiPro/LpSS/blob/master/result/Procedures.png)

* Weight distribution of a neuron with respect of p  

![Weight distribution of a neuron with respect of p](https://github.com/WilliamLiPro/LpSS/blob/master/result/weight_with_p.jpg)

* Summarization of test accuracy  


Dataset |    | nw    | Epochs| s=0.5 | s=0.8 | s=0.9
----  | ---- | ----  | ----  | ----  | ----  | ---- 
UCI machine learning | DNA  | 936195 | 40 | 0.9533 | 0.9233 | 0.9755 
|  | Mushroom| 90562 | 20   | 1.0000 | 0.9890 | 0.7172 
|  | Climate | 76738 | 20   | 0.9714 | 0.9643 | 0.9214 
MNIST |      | 170298| 40   |0.9957 | 0.9953 | 0.9946 
Fashion-MNIST| | 267514| 50   | 0.9436 | 0.9391 | 0.9309 
CIFAR-10 |   | 1154074| 250 | 0.9423 | 0.9383 | 0.9310 
Tiny ImageNet|  | 4644120 | 250 | 0.6377 | 0.6295 | 0.6042

# License
Apache-2.0
