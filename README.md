# Score-based-ECG-Denoising
This repository contains the codes for [DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal](https://ieeexplore.ieee.org/document/10018543)



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

H. Li, G. Ditzler, J. Roveda and A. Li, "DeScoD-ECG: Deep Score-Based Diffusion Model for ECG Baseline Wander and Noise Removal," in IEEE Journal of Biomedical and Health Informatics, doi: 10.1109/JBHI.2023.3237712.


## Acknowledgement

The data preprocessing is directly taken from [DeepFilter](https://www.sciencedirect.com/science/article/pii/S1746809421005899).


