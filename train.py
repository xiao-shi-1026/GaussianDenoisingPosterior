import torch
from data.dataset import ImageDataset
from torch.utils.data import DataLoader
import yaml
import os
from train.optimizer import create_optimizer, create_scheduler, create_loss_function
from data.blurring import corrupt
import torchvision.transforms as transforms
from data.utils import addnoise
from data.PSNR import evaluate_psnr_torch
import numpy as np
import random
from tqdm import tqdm 
if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "config.yaml")
    config = yaml.safe_load(open("config.yaml"))
    config_dict = {
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'epochs' : config["hyperparameter"]["epochs"],
        'num_workers' : config["num_workers"]["num"],
        'net': config["net"]["name"],
        # path
        'train_path' : os.path.join(config["path"]["input"], "train"),
        'validate_path' : os.path.join(config["path"]["input"], "validation"),
        'output_path' : config["path"]["output"],
        # batch set
        'train_batch_size' : config["hyperparameter"]["train_batch"],
        'validate_batch_size' : config["hyperparameter"]["vali_batch"],
        # optimizer
        'optimizer' : config["optimizer"]["name"],
        'learning_rate' : config["optimizer"]["learning_rate"],
        # loss
        'loss' : config["loss"]["name"],
        # scheduler
        'scheduler' : config["scheduler"]["name"],
        'step_size' : config["scheduler"]["step_size"],
        'gamma' : config["scheduler"]["gamma"],
        # random seed
        'seed' : config["seed"]["number"],
        # subproblem
        'subproblem': config["subproblem"]["name"],
        # server
        'server': config["server"]['status'],
        # augmentation
        'augmentation': config["augmentation"]
    }
    seed = config_dict["seed"]

    if seed is None:
        seed = random.randint(1, 10000)

    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if config_dict['augmentation'] == "YES":
        data_augmentation = transforms.RandomChoice([
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(90),
            transforms.Compose([transforms.RandomRotation(90), transforms.RandomVerticalFlip(p=1.0)]),
            transforms.RandomRotation(180),
            transforms.Compose([transforms.RandomRotation(180), transforms.RandomVerticalFlip(p=1.0)]),
            transforms.RandomRotation(270),
            transforms.Compose([transforms.RandomRotation(270), transforms.RandomVerticalFlip(p=1.0)]),
        ])
    else:
        data_augmentation = None

    train_dataset = ImageDataset(config_dict['train_path'], data_augmentation)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config_dict['train_batch_size'], shuffle=True, num_workers=config_dict['num_workers'])
    test_dataset = ImageDataset(config_dict['validate_path'], data_augmentation)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config_dict['validate_batch_size'], shuffle=False, num_workers=config_dict['num_workers'])
    
    if config_dict['net'] == 'DnCNN':
        from model.DnCNN import DnCNN
        print('Training DnCNN')
        model = DnCNN(channels = 3)
    elif config_dict['net'] == 'UNet':
        from model.UNet import UNet
        print('Training UNet')
        model = UNet(in_channels=3, out_channels=3)
    
    num_gpus = torch.cuda.device_count()

    criterion = create_loss_function(config_dict['loss'])
    optimizer = create_optimizer(model, config_dict['optimizer'], config_dict['learning_rate'])
    scheduler = create_scheduler(optimizer, config_dict['scheduler'], config_dict['step_size'], config_dict['gamma'])
    
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for training...")
        model = torch.nn.DataParallel(model)
    
    model = model.to(config_dict['device'])
    
    num_epochs = config_dict['epochs']

    best_val_loss = float('inf')

    if config_dict['subproblem'] == 'IP':
        from data.inpaint import build_inpaint_freeform
        class Opt:
            image_size = 256
            device = config_dict['device']

        opt = Opt()
        mask_type = 'freeform1020' 
        inpaint_fn = build_inpaint_freeform(opt, mask_type)

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")

    # Training Loop
    model.train()
    running_loss = 0.0
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.6f}")

    train_iter = train_loader
    if config_dict['server'] == 'NO':
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", total=len(train_loader))

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(config_dict['device']), labels.to(config_dict['device'])
        if config_dict['subproblem'] == 'IP':
            inputs, _ = inpaint_fn(inputs)
            inputs = inputs.to(config_dict['device'])
        inputs = addnoise(inputs, [5, 5], config_dict['device'])
        outputs = model(inputs)
        loss = criterion(outputs, labels) / (inputs.size()[0]*2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if config_dict['server'] == 'NO':
            train_iter.set_postfix(loss=loss.item())
            train_iter.update(1)
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Training Completed. Average Loss: {avg_train_loss:.4f}")
    torch.cuda.empty_cache()

    # Validation Loop
    model.eval()
    val_loss = 0.0

    val_iter = test_loader
    if config_dict['server'] == 'NO':
        val_iter = tqdm(test_loader, desc=f"Epoch {epoch + 1} Validation", total=len(test_loader))

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(config_dict['device']), labels.to(config_dict['device'])
            if config_dict['subproblem'] == 'IP':
                inputs, _ = inpaint_fn(inputs)
                inputs = inputs.to(config_dict['device'])
            inputs = addnoise(inputs, [5, 5], config_dict['device'])
            outputs = model(inputs)
            loss = criterion(outputs, labels) / (inputs.size()[0]*2)
            val_loss += loss.item()
            if config_dict['server'] == 'NO':
                val_iter.update(1)
    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch {epoch + 1} Validation Completed. Average Loss: {avg_val_loss:.4f}")
    torch.cuda.empty_cache()

    avg_psnr = evaluate_psnr_torch(test_loader, model, config_dict['device'])
    print(f"Epoch {epoch + 1} Average PSNR: {avg_psnr:.4f}")

    # Save the model if validation loss is the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        if num_gpus > 1:

            torch.save(model.module.state_dict(), os.path.join(config_dict["output_path"], "inpainting_unet.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(config_dict["output_path"], "inpainting_unet.pth"))

        print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    # Step the scheduler
    scheduler.step()

print("Training complete. Best validation loss:", best_val_loss)
