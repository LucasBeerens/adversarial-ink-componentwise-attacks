import numpy as np


def specific(net, data, att):
    """
    Perform targeted attacks on a dataset. Include all possible targets.

    Args:
    - net: The neural network model.
    - data: The dataset.
    - att: The attack function to be applied.

    Returns:
    - The results of the specific attacks.
    """
    outputSize = net(data[0][0]).size(-1)
    results = np.zeros((len(data) * (outputSize - 1), 1))

    for i, data in enumerate(data):
        img, label = data

        for c in range(outputSize-1):
            target = c + (c >= label)
            _, eps = att(net, img, target)
            results[i*(outputSize-1)+c] = eps

    return results


def any(net, data, att):
    """
    Perform untargeted attacks on a dataset.

    Args:
    - net: The neural network model.
    - data: The dataset.
    - att: The attack function to be applied.

    Returns:
    - The results of the attacks.
    """
    results = np.zeros((len(data), 1))

    for i, data in enumerate(data):
        img, _ = data
        _, eps = att(net, img)
        results[i] = eps

    return results
