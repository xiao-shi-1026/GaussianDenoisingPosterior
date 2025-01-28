import torch
# from model.DnCNN import DnCNN
from model.DnCNN import TestDnCNN
from data.dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
from train.optimizer import create_optimizer, create_scheduler, create_loss_function
from data.utils import addnoise


if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "config.yaml")
    config = yaml.safe_load(open("config.yaml"))
    config_dict = {
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'epochs' : config["hyperparameter"]["epochs"],
        'num_workers' : config["num_workers"]["num"],
        'noise_level' : [config["noise_level"]['start'], config["noise_level"]['end']],
        'net_mode': config["net_mode"]["mode"],
        # path
        'train_path' : os.path.join(config["path"]["input"], "train"),
        'validate_path' : os.path.join(config["path"]["input"], "validation"),
        'output_path' : config["path"]["output"],
        # batch set
        'train_crop_size' : config["hyperparameter"]["train_crop"],
        'validate_crop_size' : config["hyperparameter"]["vali_crop"],
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
    }

    train_dataset = ImageDataset(config_dict['train_path'], config_dict['device'], config_dict['train_crop_size'], config_dict['net_mode'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=config_dict['train_batch_size'], shuffle=True, num_workers=config_dict['num_workers'])
    test_dataset = ImageDataset(config_dict['validate_path'], config_dict['device'], config_dict['validate_crop_size'], config_dict['net_mode'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=config_dict['validate_batch_size'], shuffle=False, num_workers=config_dict['num_workers'])

    # model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='BR')
    model = TestDnCNN(channels = 3)
    criterion = create_loss_function(config_dict['loss'])
    optimizer = create_optimizer(model, config_dict['optimizer'], config_dict['learning_rate'])
    scheduler = create_scheduler(optimizer, config_dict['scheduler'], config_dict['step_size'], config_dict['gamma'])
    model = model.to(config_dict['device'])
    
    num_epochs = config_dict['epochs']

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        # Training Loop
        model.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

        with tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} (Training)", dynamic_ncols=True) as progress_bar:
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(config_dict['device']), labels.to(config_dict['device'])
                inputs = addnoise(inputs, config_dict['noise_level'], config_dict['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels) / (inputs.size()[0]*2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item(), "LR": f"{current_lr:.6f}"})

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Training Completed. Average Loss: {avg_train_loss:.4f}")
        torch.cuda.empty_cache()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)", dynamic_ncols=True) as progress_bar:
                for batch_idx, (inputs, labels) in enumerate(progress_bar):
                    inputs, labels = inputs.to(config_dict['device']), labels.to(config_dict['device'])
                    inputs = addnoise(inputs, config_dict['noise_level'], config_dict['device'])
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / (inputs.size()[0]*2)
                    val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch + 1} Validation Completed. Average Loss: {avg_val_loss:.4f}")
        torch.cuda.empty_cache()
        # Save the model if validation loss is the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config_dict["output_path"], "deblurring.pth"))
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # Step the scheduler
        scheduler.step()

    print("Training complete. Best validation loss:", best_val_loss)
