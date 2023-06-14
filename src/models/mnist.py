import torch.nn as nn
import torch.nn.functional as F
import torch


class MNIST1(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the MNIST1 neural network model.
        This is a single fully connected layer.
        """
        super(MNIST1, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        """
        Perform the forward pass of the MNIST1 model.

        Args:
        - x: The input image.
        """
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class MNIST2(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the MNIST2 neural network model.
        This consists of two fully connected layers with tanh activation.
        """
        super(MNIST2, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        """
        Perform the forward pass of the MNIST2 model.

        Args:
        - x: The input image.
        """
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class MNIST3(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the MNIST3 neural network model.
        This consists of two fully connected layers with ReLu activation.
        """
        super(MNIST3, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        """
        Perform the forward pass of the MNIST3 model.

        Args:
        - x: The input image.
        """
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNIST4(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the MNIST4 neural network model.
        This is a convolution neural network.
        """
        super(MNIST4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,14,14)
        )
        self.conv2 = nn.Sequential(  # (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,7,7)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        """
        Perform the forward pass of the MNIST4 model.

        Args:
        - x: The input image.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x
