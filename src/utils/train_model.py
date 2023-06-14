import torch.nn as nn
import torch.optim as optim


def train_model(net, dataloader, epochs, lr=0.001):
    """
    Train a neural network model on a training dataset.
    Prints the training loss for every 100 mini-batches and
    prints "Training Complete" after training is finished.

    Args:
    - net: The neural network model.
    - dataloader: The training dataset loader.
    - epochs: Number of epochs to train.
    - lr: Learning rate for the optimizer.

    Returns:
    - None
    """
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                print('[%d, %d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Training Complete')
