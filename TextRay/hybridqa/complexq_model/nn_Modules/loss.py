import torch.nn as nn
import torch.nn.functional as F
import torch
from cuda import cuda_wrapper

class Dynamic_Cross_Entropy_Loss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input, target, weights=None):
        if weights is None:
            loss = F.cross_entropy(input, target)
        else:
            raw_loss = F.cross_entropy(input, target, reduction='none')
            loss = torch.mul(weights, raw_loss).mean()

        return loss


if __name__ == '__main__':
    input = cuda_wrapper(torch.tensor([[0.1, 0.9], [0.2, 0.8]]))
    target = cuda_wrapper(torch.tensor([1, 0]))
    weights = cuda_wrapper(torch.tensor([2, 0])).float()
    loss_instance = Dynamic_Cross_Entropy_Loss()
    print loss_instance(input, target)
    print loss_instance(input, target, weights)
    weights = cuda_wrapper(torch.tensor([1, 1])).float()
    print loss_instance(input, target, weights)
    weights = cuda_wrapper(torch.tensor([0, 2])).float()
    print loss_instance(input, target, weights)

