# DBD-Net
This repository is the `pytorch` code for our paper `"DBD-Net: A Dual-Branch Decoding Network for Noisy Image Compressed Sensing"`.  
For any questions, feel free to contact us: shen65536@mail.nwpu.edu.cn  
## 1. Introduction ##
**1) Settings**  

We adopt the [`BSDS500`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) as the training set. The tests (in channel-by-channel manner) are conducted using the [`Set5`](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [`Set14`](https://huggingface.co/datasets/eugenesiow/Set14), [`CBSD68`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [`Manga109`](http://www.manga109.org/en/) datasets. DBD-Net is trained for 100 epochs with a batch size of 32. The optimization algorithm employed for training is Adam with a schedule of learning rates. The learning rate is set as 10−3 for the first 50 epochs, 10−4 for epochs 51 to 75, and 10−5 for the final 25 epochs. We conduct model preservation via verification on [`Set11`](https://github.com/KuldeepKulkarni/ReconNet) after every training epoch. Our method is implemented on a Ubuntu platform with the PyTorch 1.9.0 framework, an Intel Core i7-11700 @ 2.50 GHz CPU and a GeForce RTX 3090 GPU with 24 GB RAM.  

**2）Project structure**
```
(DBD-Net)
|-dataset
|    |-train  
|        |-BSDS500 (.jpg)  
|    |-val  
|        |-Set11  
|            |-r (.png)  
|            |-g (.png)  
|            |-b (.png) 
|            |-(.png)  
|    |-test  
|        |-Set5  
|            |-r (.png)  
|            |-g (.png)  
|            |-b (.png) 
|            |-(.png)  
|        |-Set14 (.png)  
|            |-... (Same as Set5)  
|        |-CBSD68 (.png)  
|            |-... (Same as Set5)  
|        |-Manga109 (.png)  
|            |-... (Same as Set5)  
|-gene_images (*Note*: This folder will appear after the testing.)
|    |-Set5
|        |-recon
|            |-... (Testing results .png)
|        |-sum.txt
|        |-details.txt
|    |-... (Testing sets)
|-models
|    |-__init__.py  
|    |-method.py  
|    |-module.py  
|-results  
|    |-1  
|    |-4  
|    |-... (Sampling rates)
|-utils 
|    |-__init__.py  
|    |-config.py  
|    |-loader.py  
|-test.py  
|-train.py
|-train.sh
```

**3) Competting methods**  

|Methods|Sources|Type|
|:----|:----|:----|
| ![DT-SPL](https://latex.codecogs.com/svg.image?\textbf{DT-SPL}) | [Data Compress. Conf.](https://ieeexplore.ieee.org/document/5453522) | Traditional algorithm |
| ![MH-SPL](https://latex.codecogs.com/svg.image?\textbf{MH-SPL}) | [Conf. Rec. Asilomar Conf. Signals Syst. Comput.](https://ieeexplore.ieee.org/document/6190204) | Traditional algorithm |
| ![GSR](https://latex.codecogs.com/svg.image?\textbf{GSR}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/6814320) | Traditional algorithm |
| ![ReconNet](https://latex.codecogs.com/svg.image?\textbf{ReconNet})| [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/7780424/) | Deep-learning Model |
| ![ISTA-Net (plus)](https://latex.codecogs.com/svg.image?\textbf{ISTA-Net}^{&plus;}) | [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/8578294) | Deep-learning Model |
| ![AMP-Net-9BM](https://latex.codecogs.com/svg.image?\textbf{AMP-Net-9BM}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/9298950) | Deep-learning Model |
| ![CSNet (plus)](https://latex.codecogs.com/svg.image?\textbf{CSNet}^{&plus;}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/8765626/) | Deep-learning Model |

**4) Performance demonstrates**  

**I.** Visual comparisons under `Gaussian noise` with variance `σ = 0.02` in the case of dataset `CBSD68`, `τ ∈ {0.01, 0.04, 0.10, 0.25}`. The results of PSNR, SSIM and LPIPS are given below the reconstructed images:  

<div align=center><img src="https://github.com/EchoSPLab/DBD-Net/blob/master/demo_images/SP.png"/></div>  

**II.** The recovery image under `salt-and-pepper noise` (the `SNR` of the first row and second row are `0.99 and 0.95`, respectively) in the case of dataset `Set5`, sampling rate `τ = 0.1`. PSNR, SSIM and LPIPS are given below the reconstructed images:  

<div align=center><img src="https://github.com/EchoSPLab/DBD-Net/blob/master/demo_images/GA.png"/></div>  

**III.** PSNR and LPIPS comparisons under `Gaussian noise` with variance `σ ∈ {0.01, 0.02, 0.05}` in the case of dataset `Manga109` at `τ = 0.01` (first row), and dataset `CBSD68` at `τ = 0.04` (second row), respectively.

<div align=center><img src="https://github.com/EchoSPLab/DBD-Net/blob/master/demo_images/boxes.png"/></div>  

## 2. Useage ##  
**1) For training DBD-Net.**  

* Put the `BSDS500 (.jpg)` folder (including training set (200), validation set (100) and test set (200)) into `./dataset/train/`.  
* e.g. If you want to train DBD-Net at sampling rate τ = 0.1 with GPU No.0, please run the following command. The train set will be automatically packaged and our DBD-Net will be trained with default parameters (Make sure you have enough GPU RAM):  
```
python train.py --device 0 --rate 0.1
```
* Also you can also run our shell script directly, it will automatically train the model at all sampling rates:  
```
sh train.sh
```
* Your trained models (.pth) will save in the `models` folder, it should contains `info.pth`, `model.pth`, `optimizer.pth` and `log.txt`, respectively represents the information during the training process, trained model parameters, optimizer information, and the reconstruction performance (PSNR, SSIM, LPIPS) of the verification set after one training epoch.  

**2) For testing DBD-Net.**  
* Put the `Set5 (.png)`, `Set14 (.png)`, `CBSD68 (.png)` and `Manga109 (.png)` folders into `./dataset/test/`.  
* For example, if you want to test DBD-Net at sampling rate τ = 0.1 with GPU No.0, please run:  
```
python test.py --device 0 --rate 0.1
```  
* For convenience of testing, this command will perform image sampling and reconstruction upon `all the test datasets` at `one sampling rate`. This is an example of the test results from the command line:  
```
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
Loading model from: /usr/local/Caskroom/miniconda/base/envs/DL/lib/python3.8/site-packages/lpips/weights/v0.1/alex.pth
r    bird.png
r    butterfly.png
r    head.png
r    woman.png
r    baby.png
g    bird.png
g    butterfly.png
g    head.png
g    woman.png
g    baby.png
b    bird.png
b    butterfly.png
b    head.png
b    woman.png
b    baby.png

Set5 test done.
```
* After that, the results of all tests will be saved to `./gene_images/`. `recon` folder includes all the reconstructed images, `sum.txt` shows the average results of the test set, `detail.txt` shows the Each result of the test set.  
## End ##  

We appreciate your reading and attention. If you want to see more results and details about our DBD-Net, please refer to our paper.  
