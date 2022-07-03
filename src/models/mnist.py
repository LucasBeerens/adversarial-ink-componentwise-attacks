import torch.nn as nn
import torch.nn.functional as F
import torch

class MNIST1(nn.Module):
    def __init__(self) -> None:
        super(MNIST1, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.softmax(self.fc(x))
        return x




class MNIST2(nn.Module):
    def __init__(self) -> None:
        super(MNIST2, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = torch.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x




class MNIST3(nn.Module):
    def __init__(self) -> None:
        super(MNIST3, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x




class MNIST4(nn.Module):
    def __init__(self) -> None:  
        super(MNIST4, self).__init__()  
        self.conv1 = nn.Sequential(  
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2) # (16,14,14)
                     )
        self.conv2 = nn.Sequential( # (16,14,14)
                     nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14)
                     nn.ReLU(),
                     nn.MaxPool2d(2) # (32,7,7)
                     )
        self.out = nn.Linear(32*7*7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # (batch, 32,7,7) -> (batch, 32*7*7)
        x = self.out(x)
        return x