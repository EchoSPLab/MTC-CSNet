# DBD-Net
This repository is the `pytorch` code for our paper `"DBD-Net: A Dual-Branch Decoding Network for Noisy Image Compressed Sensing"`.  
For any questions, feel free to contact us: shen65536@mail.nwpu.edu.cn  
## 1. Introduction ##
**1) Settings**  

We adopt the [`BSDS500`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) as the training set. The tests are conducted using the [`Set5`](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [`Set14`](https://huggingface.co/datasets/eugenesiow/Set14), [`CBSD68`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [`Manga109`](http://www.manga109.org/en/) datasets. DBD-Net is trained for 100 epochs with a batch size of 32. The optimization algorithm employed for training is Adam with a schedule of learning rates. The learning rate is set as 10−3 for the first 50 epochs, 10−4 for epochs 51 to 75, and 10−5 for the final 25 epochs. We conduct model preservation via verification on [`Set11`](https://github.com/KuldeepKulkarni/ReconNet) after every training epoch. Our method is implemented on a platform with the PyTorch 1.9.0 framework, an Intel Core i7-11700 @ 2.50 GHz CPU and a GeForce RTX 3090 GPU with 24 GB RAM.  

**2）Project structure**
```
DBD-Net
|-dataset
|    |-train  
|        |-BSDS500 (.jpg)  
|    |-val  
|        |-Set11 (.png)  
|            |-r (.png)  
|            |-g (.png)  
|            |-b (.png) 
|    |-test  
|        |-Set5 (.png)  
|            |-r (.png)  
|            |-g (.png)  
|            |-b (.png)  
|        |-Set14 (.png)  
|            |-... (Same as Set5)  
|        |-BSDS100 (.png)  
|            |-... (Same as Set5)  
|        |-Urban100 (.png)  
|            |-... (Same as Set5)  
|-models
|    |-__init__.py  
|    |-method.py  
|    |-module.py  
|-results  
|    |-4  
|    |-10  
|    |-25  
|    |-... (sampling rates)
|-utils 
|    |-__init__.py  
|    |-config.py  
|    |-loader.py  
|-eval.py  
|-train.py
```

**3) Competting methods**  

|Methods|Sources|Type|
|:----|:----|:----|
| ![DT-SPL](https://latex.codecogs.com/svg.image?\textbf{DT-SPL}) | [Data Compress. Conf.](https://ieeexplore.ieee.org/document/5453522) | Traditional algorithm |
| ![MH-SPL](https://latex.codecogs.com/svg.image?\textbf{MH-SPL}) | [Conf. Rec. Asilomar Conf. Signals Syst. Comput.](https://ieeexplore.ieee.org/document/6190204) | Traditional algorithm |
| ![GSR](https://latex.codecogs.com/svg.image?\textbf{GSR}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/6814320) | Traditional algorithm |
| ![ReconNet](https://latex.codecogs.com/svg.image?\textbf{ReconNet})| [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/7780424/) | Deep-learning Model |
| ![ISTA-Net (plus)](https://latex.codecogs.com/svg.image?\textbf{ISTA-Net}^{&plus;}) | [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/8578294) | Deep-learning Model |
| ![CSNet (plus)](https://latex.codecogs.com/svg.image?\textbf{CSNet}^{&plus;}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/8765626/) | Deep-learning Model |
| ![AMP-Net-9BM](https://latex.codecogs.com/svg.image?\textbf{AMP-Net-9BM}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/9298950) | Deep-learning Model |

**4) Performance demonstrates**  

**I.** Visual comparisons under `Gaussian noise` with variance `σ = 0.02` in the case of dataset `CBSD68`, `τ ∈ {0.01, 0.04, 0.10, 0.25}`. The results of PSNR, SSIM and LPIPS are given below the reconstructed images:  

<div align=center><img src="https://github.com/EchoSPLab/DBD-Net/blob/master/demo_images/SP.png"/></div>  

**II.** The recovery image under `salt-and-pepper noise` (the `SNR` of the first row and second row are `0.99 and 0.95`, respectively) in the case of dataset `Set5`, sampling rate `τ = 0.1`. PSNR, SSIM and LPIPS are given below the reconstructed images:  

<div align=center><img src="https://github.com/EchoSPLab/DBD-Net/blob/master/demo_images/GA.png"/></div>  

**III.** PSNR and LPIPS comparisons under `Gaussian noise` with variance `σ ∈ {0.01, 0.02, 0.05}` in the case of dataset `Manga109` at `τ = 0.01` (first row), and dataset `CBSD68` at `τ = 0.04` (second row), respectively.

<div align=center><img src="https://github.com/EchoSPLab/DBD-Net/blob/master/demo_images/boxes.png"/></div>  

## 2. Useage ##
