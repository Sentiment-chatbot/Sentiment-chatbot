import random
import os
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(SEED)
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