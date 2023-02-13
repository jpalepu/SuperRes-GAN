# SuperRes-GAN Tensorflow:
Implementation of SRGAN to convert single image lower resolution into an upscaled version with features being preserved. 
Dataset used (for tensorflow implementation): 
MIRFLICKR-25000 (can be downlaoded from https://press.liacs.nl/mirflickr/mirdownload.html) 

Details about implementation (using tensorflow): 

The input file is 25x25 (the downlscaled version) 
The generated file by the generator is 4x Upscalled version (100 x 100)

The network is trained on 3000 image samples for 20 epochs on apple sillicon CPU

<img width="783" alt="Screenshot 2023-02-13 at 20 16 25" src="https://user-images.githubusercontent.com/44967770/218552918-71e60a93-4e04-4440-9e7c-614600c34ada.png">
