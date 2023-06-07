import torch
import numpy as np


def normwise(net, img):
    with torch.no_grad():
        output = net(img).numpy().transpose()
        classCount = output.size
        imgLen = img.nelement()

        jac = torch.autograd.functional.jacobian(net, img)
        jac = jac.view(classCount, imgLen)

        outputNorm = np.linalg.norm(output)
        jacNorm = np.linalg.norm(jac, ord=2)
        imNorm = np.linalg.norm(img)
        return jacNorm * imNorm / outputNorm


def componentwise(net, img, tol):
    with torch.no_grad():
        output = net(img).numpy().transpose()
        classCount = output.size
        imgLen = img.nelement()

        jac = torch.autograd.functional.jacobian(net, img)
        jac = jac.view(classCount, imgLen)

        B = np.abs(jac) @ np.abs(tol)
        output = net(img).numpy()
        return np.linalg.norm(B, ord=np.inf) / np.linalg.norm(output, ord=np.inf)
