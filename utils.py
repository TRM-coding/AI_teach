import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import streamlit as st
from torch.utils.data import DataLoader

# class GPUDataLoader:
#     """数据加载器，一次性将所有数据加载到GPU"""
#     def __init__(self, dataset, batch_size, device):
#         self.batch_size = batch_size
#         self.device = device
        
#         # 将所有数据加载到GPU
#         all_data = []
#         all_labels = []
#         for data, label in dataset:
#             all_data.append(data)
#             all_labels.append(label)
        
#         self.data = torch.stack(all_data).to(device)
#         self.labels = torch.tensor(all_labels).to(device)
#         self.num_samples = len(self.labels)
        
#     def __len__(self):
#         return (self.num_samples + self.batch_size - 1) // self.batch_size
    
#     def __iter__(self):
#         indices = torch.randperm(self.num_samples, device=self.device)
#         for i in range(0, self.num_samples, self.batch_size):
#             batch_indices = indices[i:i + self.batch_size]
#             yield self.data[batch_indices], self.labels[batch_indices]

def load_data(batch_size=64):
    """加载MNIST数据集并返回GPU数据加载器"""
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    st.write("load_data")
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Use standard DataLoader to load from disk (CPU) then move all tensors to device
    cpu_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    cpu_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_data=[(x.to(device), y.to(device)) for x, y in cpu_train_loader]
    test_data=[(x.to(device), y.to(device)) for x, y in cpu_test_loader]

    
    return train_data, test_data, device,len(train_dataset),len(test_dataset)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据已经在GPU上，不需要再移动
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # 数据已经在GPU上
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs, device, progress_callback=None):
    """完整的训练循环"""
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 在测试集上评估
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 如果提供了回调函数，调用它来更新UI
        if progress_callback:
            progress_callback(epoch + 1, train_loss, train_acc, test_loss, test_acc)
    
    return history
