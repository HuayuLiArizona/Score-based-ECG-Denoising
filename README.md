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

@ARTICLE{10018543,
  author={Li, Huayu and Ditzler, Gregory and Roveda, Janet and Li, Ao},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/JBHI.2023.3237712}}

## Acknowledgement

The data preprocessing is directly taken from [DeepFilter](https://www.sciencedirect.com/science/article/pii/S1746809421005899).


