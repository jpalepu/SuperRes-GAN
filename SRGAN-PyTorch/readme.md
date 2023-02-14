# SRGAN-PyTorch
## Overview
In this notebook, I re-implement the architecture of  Super-Resolution GAN (SRGAN), a GAN that enhances the resolution of images by 4x, proposed in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) in PyTorch framework.Then I train the model by using [STL10 dataset](https://cs.stanford.edu/~acoates/stl10). Finally, I used benchmarks provided by the paper to verify the results of our re-implement model.

## training
The network is trained on 5000 image samples for 1 epoch on Google colab

## Result
In the following table is my result of model

| STL10 | Scale |      SRResNet      |       SRGAN        |
|:----: |:-----:|:------------------:|:------------------:|
| PSNR  |   4   |  70.0636  |  70.6417  |
| SSIM  |   4   | 0.8246 | 0.8348 |
