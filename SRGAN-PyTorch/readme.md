# SRGAN-PyTorch
## Overview
I reimplement of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802v5) by PyTorch framework.

## training
The network is trained on 5000 image samples for 1 epoch on Google colab

## Result
In the following table is my result of model

| STL10 | Scale |      SRResNet      |       SRGAN        |
|:----: |:-----:|:------------------:|:------------------:|
| PSNR  |   4   |  70.0636  |  70.6417  |
| SSIM  |   4   | 0.8246 | 0.8348 |
