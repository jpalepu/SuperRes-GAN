# SuperRes-GAN with Tensorflow:
Implementation of SRGAN to convert single image lower resolution into an upscaled version with features being preserved. 

Dataset used (for tensorflow implementation): 
MIRFLICKR-25000 (can be downlaoded from https://press.liacs.nl/mirflickr/mirdownload.html) 

Details about implementation (using tensorflow): 

The input file is 25x25 (the downlscaled version) 
The generated file by the generator is 4x Upscalled version (100 x 100)

The network is trained on 3000 image samples for 20 epochs on apple sillicon CPU

Sample output:
<img width="783" alt="Screenshot 2023-02-13 at 20 16 25" src="https://user-images.githubusercontent.com/44967770/218741183-369af1a8-db0a-4737-95e4-9efab9b59496.png">

The implementation is strictly based on the attached paper. 
libraries used:

1.keras
2.tensorflow
3.PIL

Instructions on running the program with terminal:
1. goto the program location (sggan.py)
2. execute python srgran.py
