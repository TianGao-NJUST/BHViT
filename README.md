# BHViT
This code is an implementation of our work "BHViT: Binarized Hybrid Vision Transformer."

[Reference]: Tian Gao, Zhiyuan Zhang, Yu Zhang, Huajun Liu, Kaijie Yin, Chengzhong Xu, and Hui Kong, [BHViT: Binarized Hybrid Vision Transformer](https://arxiv.org/abs/2503.02394), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025


----------------------------------------------------------------------------------------------------------------------------------------------------------
# [BHViT: Binarized Hybrid Vision Transformer](https://arxiv.org/abs/2503.02394)
### Introduction
Model binarization has made significant progress in enabling real-time and energy-efficient computation for convolutional neural networks (CNN), offering a potential solution to the deployment challenges faced by Vision Transformers (ViTs) on edge devices. However, due to the structural differences between CNN and Transformer architectures, simply applying binary CNN strategies to the ViT models will lead to a significant performance drop. To tackle this challenge, we propose BHViT, a binarization-friendly hybrid ViT architecture and its full binarization model with the guidance of three important observations. Initially, BHViT utilizes the local information interaction and hierarchical feature aggregation technique from coarse to fine levels to address redundant computations stemming from excessive tokens. Then, a novel module based on shift operations is proposed to enhance the performance of the binary Multilayer Perceptron (MLP) module without significantly increasing computational overhead. In addition, an innovative attention matrix binarization method based on quantization decomposition is proposed to evaluate the token's importance in the binarized attention matrix. Finally, we propose a regularization loss to address the inadequate optimization caused by the incompatibility between the weight oscillation in the binary layers and the Adam Optimizer. Extensive experimental results demonstrate that our proposed algorithm achieves SOTA performance among binary ViT methods.
### Environment and Dependencies
Our code was tested with Python 3.11.7, Pytorch 2.5.1,and cuda 12.1  

Required python packages：
* PyTorch (version 2.5.1)
* numpy
* timm
* transformers 4.39.2
### Tips
   Any problem, please contact the first author (Email: gaotian970228@njust.edu.cn).

   Our pre-trained model can be downloaded at the following link
   * BHViT-small
     [somefp 70.1](https://drive.google.com/drive/folders/1K8W9LjFQIemG6Cc6xMXzmAOTgBuN9_8h)
     [binary68.4](https://drive.google.com/drive/folders/1K8W9LjFQIemG6Cc6xMXzmAOTgBuN9_8h)
   * BHViT-tiny
     [somefp 66.0](https://drive.google.com/drive/folders/1tuEdd8xkLSuwoordYdEl4VpKLRmP3xJO)
     [binary64.0](https://drive.google.com/drive/folders/1tuEdd8xkLSuwoordYdEl4VpKLRmP3xJO)
### Citation
If you find this work useful, please consider citing:
    @inproceedings{gao2025bhvit,
      title={BHViT: Binarized Hybrid Vision Transformer}, 
      author={Tian Gao and Zhiyuan Zhang and Yu Zhang and Huajun Liu and Kaijie Yin and Chengzhong Xu and Hui Kong},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
          }
### License
Our code is released under the MIT License (see LICENSE file for details).
### Acknowledgement
Our code refers to binaryViT(https://github.com/Phuoc-Hoan-Le/BinaryViT) and DeiT(https://github.com/facebookresearch/deit).
### Reference
* Le P H C, Li X. Binaryvit: Pushing binary vision transformers towards convolutional models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition workshop. 2023: 4665-4674.
