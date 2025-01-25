import torch.optim as optim

def get_optimizer(model, optimizer_name="adam", learning_rate=0.001):
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name="step_lr", step_size=10, gamma=0.1):
    if scheduler_name == "step_lr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")