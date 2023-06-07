import numpy as np


def specific(net, data, att):
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
    results = np.zeros((len(data), 1))
    for i, data in enumerate(data):
        img, _ = data
        _, eps = att(net, img)
        results[i] = eps
    return results
