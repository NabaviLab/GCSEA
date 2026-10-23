import random
import numpy as np
import torch
import yaml

def load_config(path):
    with open(path) as f: return yaml.safe_load(f)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def to_device(sample,device):
    return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in sample.items()}
