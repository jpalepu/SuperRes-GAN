# SRGAN-PyTorch
## Overview
In this notebook, I re-implement the architecture of  Super-Resolution GAN (SRGAN), a GAN that enhances the resolution of images by 4x, proposed in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) in PyTorch framework.Then I train the model by using [STL10 dataset](https://cs.stanford.edu/~acoates/stl10). Finally, I used benchmarks provided by the paper to verify the results of our re-implement model.

## Implementation Details
In this project I choosed [STL10 Dataset](https://cs.stanford.edu/~acoates/stl10/)  as training and testing Dataset.  Both of networks(Generator network and Discriminator network) are trained on 5000 image samples for 1 epoch on Google colab online computing unit (Sorry for the hardware limitation). Following is a sample result：


|#|images|description|
|---|---|----
|1| ![image](https://user-images.githubusercontent.com/50286429/218882335-4cfeff6a-740d-4082-a296-e0b902a90493.png)| **input(lr)**|
|2| ![image](https://user-images.githubusercontent.com/50286429/218882345-83ab601e-f261-4676-84be-cb57a5fa1b45.png)| **output(SRResnt)**|
|3| ![image](https://user-images.githubusercontent.com/50286429/218882362-8e7da44d-b330-40b5-9abe-e18ffb11634f.png)| **output(SRGAN)**|
|4| ![image](https://user-images.githubusercontent.com/50286429/218882380-163fbe85-67de-4243-abcd-d1bcaa35ba2b.png)| **original(hr)**|







## Result
Because of hardware limitations, I did not train the model parameters to the optimal. However, from the current results, metrics may be better by increasing the training times.
In the following table is benchmarks metrics of my model

| STL10 | Scale |      SRResNet      |       SRGAN        |
|:----: |:-----:|:------------------:|:------------------:|
| PSNR  |   4   |  70.0636  |  70.6417  |
| SSIM  |   4   | 0.8246 | 0.8348 |
