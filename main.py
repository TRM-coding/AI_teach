from model import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist_data(batch_size=64):
    """下载并加载MNIST数据集"""
    # 定义数据转换：转为Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 下载训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def evaluate(model, test_loader):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to('cpu'), y.to('cpu')
            out = model(x)
            _, predicted = torch.max(out, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    model.train()
    return accuracy

def train(model, train_loader, test_loader=None, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to('cpu'), y.to('cpu')

            # 前向传播
            out = model(x)
            loss = F.cross_entropy(out, y)

            # 反向传播与更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss = {avg_loss:.4f}", end="")
        
        
    accuracy = evaluate(model, test_loader)
    print(f", Test Accuracy = {accuracy:.2f}%")
        

if __name__ == "__main__":
    # 1. 下载并加载MNIST数据集
    print("正在下载并加载MNIST数据集...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}\n")
    
    # 2. 创建模型
    DNN = SimpleDNN()
    print("模型结构:")
    print(DNN)
    print()
    
    # 3. 训练模型
    print("开始训练...")
    train(DNN, train_loader, test_loader, epochs=10, lr=1e-3)
    
    # 4. 保存模型
    torch.save(DNN.state_dict(), 'mnist_model.pth')
    print("\n模型已保存到 mnist_model.pth")

