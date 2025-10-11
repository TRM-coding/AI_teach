import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import math
import gc
import streamlit as st

#Enable garbage collection
gc.enable()



class SimpleDNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDNN, self).__init__()
        # 输入：1 x 28 x 28 -> 展平为 784 维
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x





def transform_image(image):
    stats = ((0.1307), (0.3081))
    my_transforms = T.Compose([

                        T.ToTensor(),
                        T.Normalize(*stats)

                        ])

    return my_transforms(image)




@st.cache
def initiate_model():

    # Initiate model
    in_channels = 1
    num_classes = 10
    model = SimpleDNN(in_channels, num_classes)
    device = torch.device('cpu')
    PATH = 'mnist-resnet.pth'
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    return model



def predict_image(img):
    
    # Convert to a batch of 1
    xb = img.unsqueeze(0)

    model = initiate_model()

    # Get predictions from model
    yb = model(xb)
    # apply softamx
    yb_soft = F.softmax(yb, dim=1)
    # Pick index with highest probability
    confidence , preds  = torch.max(yb_soft, dim=1)
    gc.collect()
    # Retrieve the class label, confidence and probabilities of all classes using sigmoid 
    return preds[0].item(), math.trunc(confidence.item()*100), torch.sigmoid(yb).detach()