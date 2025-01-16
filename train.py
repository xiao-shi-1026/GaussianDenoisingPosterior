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
    train_dataset = ImageDataset(train_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    test_dataset = ImageDataset(test_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='BR')
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    num_epochs = 5

    for epoch in range(num_epochs):
        print('start' + str(epoch) +'epochs')
        model.train()  # 设置为训练模式
        running_loss = 0.0
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                print('pass1')
                inputs, labels = inputs.to(device), labels.to(device)
                # inputs = corrupt(inputs, device).to(device)
                # 前向传播
                print(inputs.shape)
                outputs = model(inputs)
                print('pass3')
                loss = criterion(outputs, labels)
                print('pass4')
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('pass5')
                running_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
                torch.cuda.empty_cache()
                print('pass6')
    print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {running_loss / len(train_loader):.4f}")

# 6. 测试模型
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the test images: {100 * correct / total:.2f}%")

