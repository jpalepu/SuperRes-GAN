# SRGAN-PyTorch
## Overview
In this notebook, I re-implement the architecture of  Super-Resolution GAN (SRGAN), a GAN that enhances the resolution of images by 4x, proposed in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) in PyTorch framework.Then I train the model by using [STL10 dataset](https://cs.stanford.edu/~acoates/stl10). Finally, I used benchmarks provided by the paper to verify the results of our re-implement model.

## Implementation Details
In this project I choosed [STL10 Dataset](https://cs.stanford.edu/~acoates/stl10/)  as training and testing Dataset.  Both of networks(Generator network and Discriminator network) are trained on 5000 image samples for 1 epoch on Google colab online computing unit (Sorry for the hardware limitation). Following is a sample result：


|#|语法|效果|
|---|---|----
|1|`![baidu](http://www.baidu.com/img/bdlogo.gif "百度logo")`|![baidu](http://www.baidu.com/img/bdlogo.gif "百度logo")
|2|`![][code-past]`|![][code-past]

**input(lr)**
![image](https://user-images.githubusercontent.com/50286429/218882335-4cfeff6a-740d-4082-a296-e0b902a90493.png)

**output(SRresnet)**
![image](https://user-images.githubusercontent.com/50286429/218882345-83ab601e-f261-4676-84be-cb57a5fa1b45.png)

**output(SRgan)**
![image](https://user-images.githubusercontent.com/50286429/218882362-8e7da44d-b330-40b5-9abe-e18ffb11634f.png)

**original(hr)**
![image](https://user-images.githubusercontent.com/50286429/218882380-163fbe85-67de-4243-abcd-d1bcaa35ba2b.png)









## Result
In the following table is my result of model

| STL10 | Scale |      SRResNet      |       SRGAN        |
|:----: |:-----:|:------------------:|:------------------:|
| PSNR  |   4   |  70.0636  |  70.6417  |
| SSIM  |   4   | 0.8246 | 0.8348 |
