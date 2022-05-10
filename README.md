# MTC-CSNet
This repository is the `pytorch` code for our paper `"MTC-CSNet: Marrying Transformer and Convolution for Image Compressed Sensing"`.  
For any questions, feel free to contact us: shen65536@mail.nwpu.edu.cn  
## 1. Introduction ##
**1) Datasets**  

Training set: [`BSDS500`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) and [`VOC2012`](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), validation set: [`Set11`](https://github.com/KuldeepKulkarni/ReconNet), testing sets: [`Set5`](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [`Set14`](https://huggingface.co/datasets/eugenesiow/Set14), [`CBSD68`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [`Manga109`](http://www.manga109.org/en/).  

**2）Project structure**
```
(MTC-CSNet)
|-dataset
|    |-train  
|        |-BSDS500 (.jpg)  
|        |-VOC2012 (.jpg)  
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
| ![MH-SPL](https://latex.codecogs.com/svg.image?\textbf{MH-SPL}) | [Conf. Rec. Asilomar Conf. Signals Syst. Comput.](https://ieeexplore.ieee.org/document/6190204) | Traditional algorithm |
| ![GSR](https://latex.codecogs.com/svg.image?\textbf{GSR}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/6814320) | Traditional algorithm |
| ![ReconNet](https://latex.codecogs.com/svg.image?\textbf{ReconNet})| [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/7780424/) | Deep-learning Model |
| ![ISTA-Net (plus)](https://latex.codecogs.com/svg.image?\textbf{ISTA-Net}^{&plus;}) | [Conf. Comput. Vis. Pattern Recog.](https://ieeexplore.ieee.org/document/8578294) | Deep-learning Model |
| ![AMP-Net](https://latex.codecogs.com/svg.image?\textbf{AMP-Net}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/9298950) | Deep-learning Model |
| ![CSNet (plus)](https://latex.codecogs.com/svg.image?\textbf{CSNet}^{&plus;}) | [Trans. Image Process.](https://ieeexplore.ieee.org/document/8765626/) | Deep-learning Model |
| ![AutoBCS](https://latex.codecogs.com/svg.image?\textbf{AutoBCS}) | [Trans. Cybern.](https://ieeexplore.ieee.org/document/9632453) | Deep-learning Model |

**4) Performance demonstrates**  

**I.** Visual comparisons under `Gaussian noise` with variance `σ = 0.02` in the case of dataset `CBSD68`, `τ ∈ {0.01, 0.04, 0.10, 0.25}`. The results of PSNR, SSIM and LPIPS are given below the reconstructed images:  

<div align=center><img src="https://github.com/EchoSPLab/DBD-Net/blob/master/demo_images/SP.png"/></div>   

**II.** PSNR and LPIPS comparisons under `Gaussian noise` with variance `σ ∈ {0.01, 0.02, 0.05}` in the case of dataset `Manga109` at `τ = 0.01` (first row), and dataset `CBSD68` at `τ = 0.04` (second row), respectively.

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
* Your trained models (.pth) will save in the `models` folder, it should contain `info.pth`, `model.pth`, `optimizer.pth` and `log.txt`, respectively represent the information during the training process, trained model parameters, optimizer information, and the reconstruction performance (PSNR, SSIM, LPIPS) of the verification set after one training epoch.  

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
