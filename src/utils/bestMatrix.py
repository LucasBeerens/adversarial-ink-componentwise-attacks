import numpy as np
from data_attacks import attackAny
import torch


def create(net, data, att):
    """
    Perform an attack on a dataset and calculate the distribution of best
    target classes for every original class.
    In most cases it is better to use analyse.table instead as that
    function can use saved date instead of having to run attacks again.
    That function also immediately puts the table in LaTeX format.

    Args:
    - net: The neural network model.
    - data: The dataset to be attacked.
    - att: The attack algorithm to be used.

    Returns:
    - The distribution of best target classes for every original class.
    """
    outputSize = net(data[0][0]).size(-1)
    results = np.zeros((outputSize, outputSize))

    # Attack the images and keep track of the resulting classifications
    alg = attackAny.Any(att)
    for entry in data:
        img, label = entry
        newImg, eps = alg(net, img)
        outputs = net(newImg)
        _, predicted = torch.max(outputs.data, 1)
        results[label, predicted] += 1

    # Calculate the distribution of best target classes for every original class.
    bestClassRatio = np.zeros((outputSize, outputSize))
    print(results)
    for i in range(outputSize):
        for j in range(outputSize):
            bestClassRatio[i, j] = results[i, j] / np.sum(results[i, :])

    return bestClassRatio
