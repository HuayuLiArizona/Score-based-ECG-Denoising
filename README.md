# Score-based-ECG-Denoising
This repository contains the codes for [DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal]()



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
