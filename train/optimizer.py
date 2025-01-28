import torch.optim as optim
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class SumSquaredError(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(SumSquaredError, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def create_optimizer(model, name, lr):
    """
    TODO: put more parameters in the optimizer.
    """

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def create_scheduler(optimizer, name, step_size, gamma):

    if name == "step_lr":
        return optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

def create_loss_function(name):

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss(size_average=False)
    elif name == "gdp_mse":
        return SumSquaredError()
    else:
        raise ValueError(f"Unsupported loss function: {name}")