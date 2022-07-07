import torchvision
import torch
import torchvision.transforms as transforms

def trainloader(bs):
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
    return torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=bs,num_workers=6)

def testloader(bs):
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
    return torch.utils.data.DataLoader(testset, shuffle=False, batch_size=bs, num_workers=6)

def testSetCorrect(net, n):
    correct = []
    with torch.no_grad():
        for data in testloader(1):
            if len(correct) >= n:
                break
            image, label = data
            outputs = net(image)
            _, predicted = torch.max(outputs.data,1)
            if predicted == label:
                correct.append(data)
    return correct