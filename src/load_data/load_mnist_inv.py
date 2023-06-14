import torchvision
import torch
import torchvision.transforms as transforms

# Define a composition of transformations to be applied to the data
transform = transforms.Compose([
    transforms.RandomInvert(p=1),
    transforms.ToTensor(),
])


def trainloader(bs):
    """
    Create a DataLoader for the training set of the MNIST dataset. 
    The transform inverts the images.

    Args:
    - bs (int): Batch size for the DataLoader.

    Returns:
    - torch.utils.data.DataLoader: DataLoader for the training set.
    """
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=bs, num_workers=6)


def testloader(bs):
    """
    Create a DataLoader for the test set of the MNIST dataset.
    The transform inverts the images.

    Args:
    - bs (int): Batch size for the DataLoader.

    Returns:
    - torch.utils.data.DataLoader: DataLoader for the test set.
    """
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, shuffle=False, batch_size=bs, num_workers=6)


def testSetCorrect(net, n):
    """
    Get a list of correctly classified samples from the test set using a given network.

    Args:
    - net: The neural network model.
    - n (int): Maximum number of correct samples to retrieve.

    Returns:
    - list: List of correctly classified samples.
    """
    correct = []
    with torch.no_grad():
        for data in testloader(1):
            if len(correct) >= n:
                break
            image, label = data
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            if predicted == label:
                correct.append(data)
    return correct
