# SuperRes-GAN with Tensorflow:
Implementation of SRGAN to convert single image lower resolution into an upscaled version with features being preserved. 

Dataset used (for tensorflow implementation): 
MIRFLICKR-25000 (can be downlaoded from https://press.liacs.nl/mirflickr/mirdownload.html) 

Details about implementation (using tensorflow): 

The input file is 25x25 (the downlscaled version) 
The generated file by the generator is 4x Upscalled version (100 x 100)

The network is trained on 3000 image samples for 20 epochs on apple sillicon CPU

The implementation is strictly based on the attached paper. 
libraries used:

1.keras
2.tensorflow
3.PIL

Instructions on running the program with terminal:
1. goto the program location (sggan.py)
2. execute python srgran.py
