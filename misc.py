import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True


def gen_folder_name(args):
    def get_attr(inst, arg):
        value = getattr(inst, arg)
        if isinstance(value, float):
            return f"{value:.4f}"
        else:
            return value
    folder_name = ''
    for arg in vars(args):
        folder_name += f'{arg}-{get_attr(args, arg)}~'
    return folder_name[:-1]


def predict_diagonal(fx):
    # Rule 1: Locate the largest value on the matrix diagonal, 
    # i.e., assuming one class prompter only being responsible for one class. 
    prob = torch.softmax(fx, -1)
    prob_d = torch.diagonal(prob, dim1=-2, dim2=-1)
    pred = torch.argmax(prob_d, -1)
    return pred