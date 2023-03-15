import numpy as np


def specific(net, data, att, name):
    imgCount = len(data)
    inputSize = data[0][0].squeeze().size()
    outputSize = net(data[0][0]).size(-1)

    resultsEps = np.zeros((len(data), outputSize))
    resultsImg = np.zeros((imgCount, outputSize, inputSize[0], inputSize[1]))
    resultsLabels = np.zeros((imgCount, 1))

    for i, dataPoint in enumerate(data):
        img, label = dataPoint
        resultsImg[i] = label
        for c in range(outputSize-1):
            target = c + (c >= label)
            resultsImg[i,c,:,:], resultsEps[i, c] = att(net, img, target)

    np.save('../resultsNumpy/{}_epss.npy'.format(name), resultsEps)
    np.save('../resultsNumpy/{}_imgs'.format(name), resultsImg)
    np.save('../resultsNumpy/{}_labels'.format(name), resultsLabels)


def any(net, data, att, name):
    imgCount = len(data)
    inputSize = data[0][0].squeeze().size()

    resultsEps = np.zeros((len(data),))
    resultsImg = np.zeros((imgCount, inputSize[0], inputSize[1]))
    resultsLabels = np.zeros((imgCount,))

    for i, dataPoint in enumerate(data):
        img, resultsLabels[i] = dataPoint
        resultsImg[i], resultsEps[i] = att(net, img)

    np.save('../resultsNumpy/{}_epss.npy'.format(name), resultsEps)
    np.save('../resultsNumpy/{}_imgs'.format(name), resultsImg)
