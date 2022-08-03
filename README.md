# Score-based-ECG-Denoising
This repository contains the codes for [DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal](https://arxiv.org/abs/2208.00542)



The deep learning models were implemented using PyTorch.


# Dataset

~~~
bash ./data/download_data.sh
~~~


# Train the model
~~~
!python -W ignore main_exp.py --n_type=1
~~~
~~~
!python -W ignore main_exp.py --n_type=2
~~~


# Evaluation the model
~~~
!python -W ignore eval_new.py
~~~



## Citing our work

our work is currantly under review.

## Acknowledgement

The data preprocessing is directly taken from [DeepFilter](https://www.sciencedirect.com/science/article/pii/S1746809421005899).


