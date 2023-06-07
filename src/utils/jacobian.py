import torch
import numpy as np


def approx(net, img, eps=1e-3):
    net.eval()
    output = net(img)
    imShape = img.size()
    imWidth = imShape[-1]
    classCount = output.nelement()
    imgLen = img.nelement()
    unitTensors = torch.zeros((imWidth, imWidth, imWidth, imWidth))
    jac = np.zeros((imgLen, classCount))
    for i in range(imWidth):
        for j in range(imWidth):
            unitTensors[i, j, i, j] = 1
            new = (
                net(img + unitTensors[i, j, :, :].view(imShape) * eps) - output)/eps
            jac[i*imWidth + j, :] = new
    return torch.transpose(torch.tensor(jac), 0, 1).float()
