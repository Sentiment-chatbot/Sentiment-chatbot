import random
import os
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from torch.backends import cudnn

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def configure_cudnn(debug):
    cudnn.enabled = True
    cudnn.benchmark = True
    if debug:
        cudnn.deterministic = True
        cudnn.benchmark = False

def get_num_workers():
    if cpu_count() > 5:
        num_workers = cpu_count() // 2
    elif cpu_count() < 2:
        num_workers = 0
    else:
        num_workers = 2
        
    return num_workers

def init_weight(module: nn.Module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()
