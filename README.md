# Deep Generative Network for Structured Deformable Mesh

![](./teaser.jpg)

## Goal

Propose a deep generative network to explore shape collections, which can be used for interpolation, generation, and editing.

## Abstract

We introduce SDM-NET, a deep generative neural network which produces structured deformable meshes. Specifically, the network is trained to generate a spatial arrangement of closed, deformable mesh parts, which respects the global part structure of a shape collection, e.g., chairs, airplanes, etc. Our key observation is that while the overall structure of a 3D shape can be complex, the shape can usually be decomposed into a set of parts, each homeomorphic to a box, and the finer-scale geometry of the part can be recovered by deforming the box. The architecture of SDM-NET is that of a two-level variational autoencoder (VAE). At the part level, a PartVAE learns a deformable model of part geometries. At the structural level, we train a Structured Parts VAE (SP-VAE), which jointly learns the part structure of a shape collection and the part geometries, ensuring the coherence between global shape structure and surface details. Through extensive experiments and comparisons with the state-of-the-art deep generative models of shapes, we demonstrate the superiority of SDM-NET in generating meshes with visual quality, flexible topology, and meaningful structures, benefiting shape interpolation and other subsequent modeling tasks.

## Description

Given a collection of shapes of the same category with part-level labels, our method represents them using a structured set of deformable boxes, each corresponding to a part. We shape collections by allowing individual boxes to be flexibly deformable and propose a two-level VAE architecture called SDM-NET, including PartVAE for encoding the geometry of deformable boxes, and SP-VAE for joint encoding of part geometry and global structure such as symmetry and support. Moreover, to ensure that decoded shapes are physically plausible and stable, we introduce an optimization based on multiple constraints including support stability, which can be compactly formulated and efficiently optimized. Our SDM-NET model allows easy generation of plausible meshes with flexible structures and fine details.

## Prerequisites

1. System

    - **Ubuntu 16.04 or later**
    - **NVIDIA GPU + CUDA 9.0 cuDNN 7.6.1**

2. Software

    - Python 3.6

        ```shell
        sh install.sh
        ```

    - MATLAB


## Data and Checkpoint

All the [data](https://drive.google.com/file/d/1myWnHmuk2XD7lyHJL7KAgok89DT7SETF/view?usp=sharing) and [checkpoint](https://drive.google.com/file/d/1ItmG9tQ7vEE31anDU_z2yER2Wdon9_Ez/view?usp=sharing) used to reproduce the result is stored in google drive. Links are also available in `data_checkpoint_link.txt`. Save the two zip files in current directory. 

After downloading the data and checkpoint, the directory tree looks like this:

```txt
── SDMNET_stamp
   ├── checkpoint
   ├── code
   ├── data
   ├── data_checkpoint_link.txt
   ├── install.sh
   ├── liability form.pdf
   ├── introduction.txt
   ├── mvdata.sh
   ├── README.md
   └── teaser.jpg
```

Then execute:
```sh
unzip data.zip
cp ./data/chair* ./code/python/chair_reproduce
cp ./data/plane* ./code/python/plane_reproduce

unzip checkpoint.zip
mv ./checkpoint/05060123_6863bin_1-joint_1-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_chair-trcet_1.0 ./code/python/chair_reproduce
mv ./checkpoint/05050238_2556bin_0-joint_0-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_plane-trcet_1.0 ./code/python/plane_reproduce
```

## Reproduce

1. Interpolation between shapes

   ```shell
   cd ./code/python/plane_reproduce
   CUDA_VISIBLE_DEVICES='' python ./test_stacknewvae.py --output_dir ./05050238_2556bin_0-joint_0-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_plane-trcet_1.0 --interids  '1f5537f4747ec847622c69c3abc6f80' 'f16381a160f20bc4a3b534252984039' 'efbb9337b9bd3cab56ed1d365b05390d'
   
   cd ../chair_reproduce
   CUDA_VISIBLE_DEVICES='' python ./test_stacknewvae.py --output_dir ./05060123_6863bin_1-joint_1-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_chair-trcet_1.0 --interids '4e664dae1bafe49f19fb4103277a6b93' '1c17cc67b8c747c3febad4f49b26ec52' '2bbf00f0c583fd8a4b3c42e318f3affc'
   ```

   After running the command, interpolated meshes are saved to a sub directory `interpolationxxxx`  in the checkpoint directory specified by `--output_dir` argument,  `./05050238_2556bin_0-joint_0-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_plane-trcet_1.0/interpolation80000`for example.

2. Postprocess

   Open `./code/matlab` in MATLAB. Execute following commands.

   ```matlab
   GetOptimizedObj('../python/plane_reproduce/05050238_2556bin_0-joint_0-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_plane-trcet_1.0/interpolation80000', 'plane', 2, 0, 0)
    
   GetOptimizedObj('../python/chair_reproduce/05060123_6863bin_1-joint_1-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_chair-trcet_1.0/interpolation100000', 'chair', 2, 0, 0)
   ```

   The output meshes are in `../python/plane_reproduce/05050238_2556bin_0-joint_0-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_plane-trcet_1.0/interpolation80000` and `../python/chair_reproduce/05060123_6863bin_1-joint_1-l0_100.0-l2_10.0-l3_1.0-l4_0.001-model_chair-trcet_1.0/interpolation100000`.

   Open those meshes in meshlab and you will see the reproduced results for **Fig.1** in the original paper. (**NOTE**: Some parts' faces might have wrong normals, use **double face** mode for beter visualization)

## Citation
If you found this code useful please cite our work as:

&nbsp;&nbsp;&nbsp;&nbsp;@article{gaosdmnet2019,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author = {Gao, Lin and Yang, Jie and Wu, Tong and Yuan, Yu-Jie and Fu, Hongbo and Lai, Yu-Kun and Zhang, Hao(Richard)},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title = {{SDM-NET}: Deep Generative Network for Structured Deformable Mesh},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH Asia 2019)},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year = {2019},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;volume = 38,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pages = {243:1--243:15},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;number = 6  
&nbsp;&nbsp;&nbsp;&nbsp;}