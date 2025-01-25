import torch
from DnCNN import DnCNN
from torch.nn.modules.loss import _Loss
import torch.optim as optim
from dataset import ImageDataset
from torch.utils.data import DataLoader
from blurry import corrupt
from tqdm import tqdm

class sum_squared_error(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\Data\train"
    test_path = r"C:\Users\sx119\Desktop\GaussianDenoisingPosterior\Data\validation"
    train_dataset = ImageDataset(train_path, device, 128)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataset = ImageDataset(test_path, device, 1024)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='BR')
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        
        # Training Loop
        model.train()  # Training mode
        running_loss = 0.0
        with tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} (Training)", dynamic_ncols=True) as progress_bar:
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
                torch.cuda.empty_cache()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Training Completed. Average Loss: {avg_train_loss:.4f}")

    model.eval()  
    total_loss = 0.0
    total_samples = 0

    sse_loss = sum_squared_error()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = sse_loss(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

    average_sse_loss = total_loss / total_samples
    print(f"Average Sum Squared Error Loss of the model on the test set: {average_sse_loss:.4f}")
