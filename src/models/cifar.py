import torch.nn as nn
import torch.nn.functional as F
import torch


class CIFAR1(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the CIFAR1 neural network model.
        This neural net consistst of two linear layers
        with tanh activation.
        """
        super(CIFAR1, self).__init__()
        self.fc1 = nn.Linear(3072, 768)
        self.fc2 = nn.Linear(768, 10)

    def forward(self, x):
        """
        Perform the forward pass of the CIFAR1 model.

        Args:
        - x: The input image.
        """
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR2(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the CIFAR2 neural network model.
        This is a convolutional neural network.
        """
        super(CIFAR2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Perform the forward pass of the CIFAR2 model.

        Args:
        - x: The input image.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
