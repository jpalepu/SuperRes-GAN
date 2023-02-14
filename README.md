# Single Image Super Resolution using GANs
## Credit
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi


Re-implement by:
 - Weihao Wang(1988339)
 - jithin kumar palepu(1988599)
## Bcakgroud
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: **how do we recover the finer texture details when we super-resolve at large upscaling factors?**
## Proposed Method
In this paper, authors presented SRGAN, a generative adversarial network (GAN) for image superresolution (SR).It allows one to train a generative model G wich is
<img width="783" alt="Screenshot 2023-02-13 at 20 16 25" src="https://github.com/https-deeplearning-ai/GANs-Public/blob/master/SRGAN-Generator.png?raw=true">
with the goal of fooling a differentiable discriminator D with following structure
<img width="783" alt="Screenshot 2023-02-13 at 20 16 25" src="https://github.com/https-deeplearning-ai/GANs-Public/blob/master/SRGAN-Discriminator.png?raw=true">
that is trained to distinguish super-resolved images from real images.
To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss.

![image width="783"](https://user-images.githubusercontent.com/50286429/218705342-d7062804-50b9-4fe5-a22c-a1ca602374d2.png)



The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images.
![image](https://user-images.githubusercontent.com/50286429/218705557-3d6aac46-5a3b-4b11-95fa-85e31164d3a5.png)

In addition, we use a VGG content loss motivated by perceptual similarity instead of similarity in pixel space.
![image](https://user-images.githubusercontent.com/50286429/218706873-ec46ed92-5b6a-4dfc-ad18-7c5109123a3a.png)


All in all, they trained their model for estimating optimal parameters to minimize the loss function they provied, which is
![image](https://user-images.githubusercontent.com/50286429/218706661-45c85a26-d489-4d1d-8aad-ca37a6ec28dd.png)








# SuperRes-GAN with Tensorflow:
Implementation of SRGAN to convert single image lower resolution into an upscaled version with features being preserved. 

Dataset used (for tensorflow implementation): 
MIRFLICKR-25000 (can be downlaoded from https://press.liacs.nl/mirflickr/mirdownload.html) 

Details about implementation (using tensorflow): 

The input file is 25x25 (the downlscaled version) 
The generated file by the generator is 4x Upscalled version (100 x 100)

The network is trained on 3000 image samples for 20 epochs on apple sillicon CPU

Here are some Outputs:
<img width="783" alt="Screenshot 2023-02-13 at 20 16 25" src="https://user-images.githubusercontent.com/44967770/218552918-71e60a93-4e04-4440-9e7c-614600c34ada.png">

The implementation is strictly based on the attached paper, and the architechture of generator is:

 

Discriminator: 
<img width="1114" alt="Screenshot 2023-02-13 at 20 25 13" src="https://user-images.githubusercontent.com/44967770/218554648-f87805db-9720-44d7-b37e-2ac423645bf7.png">

Further details are explained in the markdowns cells of the implementation. 

