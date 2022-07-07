import numpy as np

def test(net, data, att):
    outputSize = net(data[0][0]).size(-1)
    results = np.zeros((len(data) * (outputSize - 1),3))
    for i, data in enumerate(data):
        img, label = data
        for c in range(outputSize-1):
            target = c + (c>=label)
            _, eps = att.attack(net,img,target)
            results[i*(outputSize-1)+c] = (label, target, eps)
    return results