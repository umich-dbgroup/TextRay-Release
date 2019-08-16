
import torch


def cuda_wrapper(var):
    if var is None:
        return None
    if torch.cuda.is_available():
        return var.cuda()
    return var


def obj_to_tensor(var, enforce_cpu=False):
    if enforce_cpu:
        return torch.tensor(var)
    if torch.cuda.is_available():
        return torch.tensor(var).cuda()
    else:
        return torch.tensor(var)
