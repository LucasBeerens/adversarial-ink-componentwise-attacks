import numpy as np
from data_attacks import attackAny
import torch


def create(net, data, att):
    outputSize = net(data[0][0]).size(-1)
    results = np.zeros((outputSize, outputSize))
    alg = attackAny.Any(att)
    for entry in data:
        img, label = entry
        newImg, eps = alg(net, img)
        outputs = net(newImg)
        _, predicted = torch.max(outputs.data, 1)
        results[label, predicted] += 1
    bestClassRatio = np.zeros((outputSize, outputSize))
    print(results)
    for i in range(outputSize):
        for j in range(outputSize):
            bestClassRatio[i, j] = results[i, j]/np.sum(results[i, :])
    return bestClassRatio
