# Lp-spherical Sparse Training

Controlling Sparsity of Neural Network with Weight Constrained on Unit Lp-Sphere.  
For more information, please contact author by: williamli_pro@163.com  

The corresponding paper is under review of Nature Machine Intelligence. 

# System Requirements
Operating systems: Windows or Linux  
Dependencies: CUDA, CuDNN, Python (version >= 3.5) Pytorch (version >= 1.2) and corresponding torchvision  

# Demo
After installed all of the requirements, please run demo.py  

# Results
![Procedures of LpSS](https://github.com/WilliamLiPro/LpSS/blob/master/result/Procedures.png)
![Weight distribution of a neuron with respect of p](https://github.com/WilliamLiPro/LpSS/blob/master/result/weight_with_p.jpg)

# References
> 1.	Han, S., Pool, J., Tran, J., & Dally, W. J. Learning both weights and connections for efficient neural networks. Neural information processing systems, 1135-1143 (2015). 
> 2.	Srinivas, S., Subramanya, A., & Babu, R. V. Training Sparse Neural Networks. In 2017 IEEE computer vision and pattern recognition (CVPR), 455-462 (IEEE, Hawaii, 2017).
> 3.	Kalchbrenner, N., Elsen, E., Simonyan, K., Noury, S. et al. Efficient Neural Audio Synthesis. In 2018 international conference on machine learning (ICML), 2410-2419 (2018).
> 4.	Elsen, E., Dukhan, M., Gale, T., & Simonyan, K. Fast Sparse ConvNets. Preprint at: https://arxiv.org/abs/1911.09723 (2019).
> 5.	Mocanu, D. C., Mocanu, E., Stone, P., Nguyen, P. H., Gibescu, M., & Liotta, A. Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science. Nat. Commun. 9, 2383 (2018).
> 6.	Thimm, G. & Fiesler, E. Evaluating pruning methods. In International Symposium on Artiﬁcial Neural Networks (1995).
> 7.	Ström, N. Sparse connection and pruning in large dynamic artiﬁcial neural networks. In European Conference on Speech Communication & Technology (1997). 
> 8.	Narang, S., Diamos G., Sengupta S., & Elsen E. Exploring sparsity in recurrent neural networks. In 5th International Conference on Learning Representations (ICLR 2017), 24-26 (2017).
> 9.	Tartaglione, E., Lepsøy, S., Fiandrotti, A., & Francini, G. Learning sparse neural networks via sensitivity-driven regularization. In proceedings of the 32nd international conference on neural information processing systems (NIPS’18), 3882–3892 (2018).
> 10.	Gale, T., Elsen, E., & Hooker, S. The state of sparsity in deep neural networks. Preprint at: http://arxiv.org/abs/1902.09574 (2019).
> 11.	Lee, N., Ajanthan, T., & Torr, P. H. S. SNIP: Single-shot network pruning based on connection sensitivity. In international conference on learning representations (2019).
> 12.	Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. Rigging the Lottery: Making All Tickets Winners. In Proceedings of Machine Learning and Systems 2020, 471-481 (2020).
> 13.	Ye, Y., Shao, Y., Deng, N., Li, C., & Hua, X. Robust Lp-norm least squares support vector regression with feature selection. Applied Mathematics and Computation 305, 32-52 (2017).
> 14.	Zhu, Q., Xu, N., Huang, S., Qian, J., & Zhang, D. Adaptive feature weighting for robust Lp-norm sparse representation with application to biometric image classification. International Journal of Machine Learning and Cybernetics 11, 463-474 (2020).
> 15.	Zhou, C., Zhang, J., & Liu, J. Lp-wgan: using lp-norm normalization to stabilize Wasserstein generative adversarial networks. Knowledge Based Systems 161, 415-424 (2018).
> 16.	Khan, N., & Stavness, I. Sparseout: Controlling Sparsity in Deep Networks. In Canadian conference on artificial intelligence, 296-307 (2019).
> 17.	Roy, S. K. , & Harandi, M. Constrained Stochastic Gradient Descent: The Good Practice. In 2017 International Conference on Digital Image Computing: Techniques and Applications (DICTA), 1-8 (2017).
> 18.	Qian N. On the momentum term in gradient descent learning algorithms. Neural Networks 12， 145-151 (1999).
> 19.	Hoyer, P. O. Non-negative matrix factorization with sparseness constraints. Journal of Machine Learning Research 5, 1457-1469 (2004).
> 20.	LeCun Y., Bottou L., Bengio Y., & Haffner P. Gradient-based learning applied to document recognition. In Proceedings of the IEEE, Vol. 86, 2278-2324 (1998).
> 21.	Xiao, H., Rasul, K., & Vollgraf, R. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Preprint at: https://arxiv.org/abs/1708.07747 (2017).
> 22.	Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. Master’s thesis (2009).
> 23.	Dua, D. & Graff, C. UCI Machine Learning Repository. http://archive.ics.uci.edu/ml (2019).
> 24.	Zagoruyko, S. & Komodakis, N. Wide residual networks. In British Machine Vision Conference, (2016).
> 25.	Xu, T., Zhou, Y., Ji, K., & Liang, Y. Convergence of SGD in learning ReLU models with separable data. Preprint at: https://arxiv.org/abs/1806.04339 (2018).
> 26.	Ioffe, S. & Szegedy, C. Batch normalization: accelerating deep network training by reducing internal covariate shift. In Proc. of the 32nd International Conference on Machine Learning, Vol. 37 of ICML’15 (ed. Bach, F. & Blei, D.), 448–456 (PMLR, 2015).
> 27.	Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. Learning Deep Features for Discriminative Localization. In computer vision and pattern recognition (CVPR), 2921-2929 (IEEE, Las Vegas, 2016).
> 28.	He, K., Zhang, X., Ren, S., & Sun, J. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. In international conference on computer vision (ICCV), 1026-1034 (2015).

