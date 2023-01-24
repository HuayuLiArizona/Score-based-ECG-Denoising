from main_model import DDPM
from denoising_model_small import ConditionalModel
import torch
import yaml


path = "config/base.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)    
    
    
path = "config/base.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)    

base_model = ConditionalModel(config['train']['feats']).to('cuda:0')
model = DDPM(base_model, config, 'cuda:0')
foldername = "./check_points/noise_type_" + str(1) + "/"
output_path = foldername + "/model.pth"

model.load_state_dict(torch.load(output_path))

# put your input here
inputs = None

num_shots = 10
outputs = 0
for i in range(num_shots):
    outputs += model.denoising(inputs, continous=False)
outputs /= num_shots