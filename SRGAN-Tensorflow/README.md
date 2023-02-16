## SuperRes-GAN with Tensorflow:

Implementation of SRGAN to convert single image lower resolution into an upscaled version with features being preserved. 

Inspired from https://github.com/jlaihong/image-super-resolution

Dataset used (for tensorflow implementation): 
MIRFLICKR-25000 (can be downlaoded from https://press.liacs.nl/mirflickr/mirdownload.html) 

Details about implementation (using tensorflow): 

The input file is 25x25 (the downlscaled version) 
The generated file by the generator is 4x Upscalled version (100 x 100)

The network is trained on 3000 image samples for 20 epochs on apple sillicon CPU

Sample output:
<img width="739" alt="Screenshot 2023-02-16 at 10 55 23" src="https://user-images.githubusercontent.com/44967770/219331866-6f7c409d-ad17-41ee-8d8f-18c473b15cc0.png">


| Metrics | SRGAN| From Paper |
| :---         |     :---:      |:---: |
| PSNR         |63.1921787      |26.44 |
| SSIM         | 0.699          |0.75  |

The implementation is strictly based on the attached paper. 
All the libraries to be installed are provided in requirements.txt

Instructions on running the program with terminal:
1. Goto the program location (main.py)
2. Execute the command: 
```
pip3 install -r requirements.txt
```
this will install all the necessary libraries used to run the program.
3. then execute the command:
```
python main.py
```
