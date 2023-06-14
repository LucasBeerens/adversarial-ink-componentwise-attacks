import torch


def test_model(net, testloader):
    """
    Test a neural network model on a test dataset.
    Prints the accuracy of the network on the test images.

    Args:
    - net: The neural network model.
    - testloader: The test dataset loader.

    Returns:
    - None
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' %
          (total, 100 * correct / total))
