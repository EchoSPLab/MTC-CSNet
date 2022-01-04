# DBD-Net
This repository is the `pytorch` code for our paper `"DBD-Net: A Dual-Branch Decoding Network for Noisy Image Compressed Sensing"`.  
For any questions, feel free to contact us: shen65536@mail.nwpu.edu.cn  
## 1. Introduction ##
**1) Settings**  

We adopt the [`BSDS500`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) as the training set. The tests are conducted using the [`Set5`](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [`Set14`](https://huggingface.co/datasets/eugenesiow/Set14), [`CBSD68`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [`Manga109`](http://www.manga109.org/en/) datasets. DBD-Net is trained for 100 epochs with a batch size of 32. The optimization algorithm employed for training is Adam with a schedule of learning rates. The learning rate is set as 10−3 for the first 50 epochs, 10−4 for epochs 51 to 75, and 10−5 for the final 25 epochs. We conduct model preservation via verification on [`Set11`](https://github.com/KuldeepKulkarni/ReconNet) after every training epoch. Our method is implemented on a platform with the PyTorch 1.9.0 framework, an Intel Core i7-11700 @ 2.50 GHz CPU and a GeForce RTX 3090 GPU with 24 GB RAM.  

**2) Competting methods**  

|Methods|Source|Type|
|:----:|:----:|:----:|
| ![DT-SPL](https://latex.codecogs.com/svg.image?\textbf{DT-SPL}) | [Data Compress. Conf.](https://ieeexplore.ieee.org/document/5453522) | Traditional algorithm |
| ![MH-SPL](https://latex.codecogs.com/svg.image?\textbf{MH-SPL}) | [Conf. Rec. Asilomar Conf. Signals Syst. Comput.](https://ieeexplore.ieee.org/document/6190204) | Traditional algorithm |
| ![GSR](https://latex.codecogs.com/svg.image?\textbf{GSR}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/6814320) | Traditional algorithm |
| ![ReconNet](https://latex.codecogs.com/svg.image?\textbf{ReconNet})| [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/7780424/) | Deep-learning Model |
| ![ISTA-Net (plus)](https://latex.codecogs.com/svg.image?\textbf{ISTA-Net}^{&plus;}) | [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/8578294) | Deep-learning Model |
| ![CSNet (plus)](https://latex.codecogs.com/svg.image?\textbf{CSNet}^{&plus;}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/8765626/) | Deep-learning Model |
| ![AMP-Net-9BM](https://latex.codecogs.com/svg.image?\textbf{AMP-Net-9BM}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/9298950) | Deep-learning Model |

**3) Performance demonstrates**  


## 2. Useage ##
Project structure:  
```

```
