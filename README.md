# Spe-TL
Sun, Y.; Liu, B.; Yu, X.; Yu, A.; Gao, K.; Ding, L. From Video to Hyperspectral: Hyperspectral Image-Level Feature Extraction with Transfer Learning. Remote Sens. 2022, 14, 5118. https://doi.org/10.3390/rs14205118

If the code is helpful to you, please give a star or fork and cite the paper. Thanks!

# Setup
The correlation layer is implemented in CUDA using CuPy, and the CuPy installation requires to correspond to the CUDA version. 

# Requirements
cupy>=5.0.0
numpy>=1.15.0
Pillow>=5.0.0
torch>=1.6.0
getopt
math
sys
scipy
tqdm
sklearn
time
datetime

# Usage

We provide a demo of the Indian Pines hyperspectral data by run the file of PWC-Net-SVM-IP-multi+vote.py. And the pre-trained PWC-Net is also provided dircectly, you can choose to re-train it on the video data[1]. Please note that due to the randomness of the parameter initialization, the experimental results might have slightly different from those reported in the paper. If you want to run the code in your own data, you can accordingly change the input and tune the parameters. Please refer to the paper for more details.
# License
Copyright (C) 2022 Yifan Sun

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.



# References
[1]  @misc{pytorch-pwc,\\
         author = {Simon Niklaus},
         title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
    }
