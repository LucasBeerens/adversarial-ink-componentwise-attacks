import numpy as np
import torch


def specific(net, img, attack, cl, n=10, clamp=False):
    """
    Scale an adversarial attack perturbation targeted at a specific class
    such that it is the smallest perturbation in that direction which
    leads to the specified classification.

    Args:
    - net: The neural network model.
    - img: The input image.
    - attack: The perturbation to be scaled.
    - cl: The target class for the attack.
    - n (int): Number of iterations for the scaling. (Doesn't do anything anymore.
                it was previously used for binary search iterations.)
    - clamp (bool): Whether to clamp the pixel values of the perturbed image.
                    This will keep the values in the interval [0,1].

    Returns:
    - The perturbed image and the relative normwise perturbation achieved.
    """
    with torch.no_grad():
        imgLen = img.nelement()
        imShape = img.size()
        x = np.reshape(img.flatten().numpy(), (imgLen, 1))
        a = 0.5
        ratio = np.linalg.norm(x) / np.linalg.norm(attack)

        for i in range(1000):
            a = i / 1000
            xNew = x + a * attack * ratio

            if clamp:
                newImg = torch.clamp(torch.tensor(
                    np.reshape(xNew, imShape)), 0, 1).float()
                d = np.linalg.norm(newImg - img) / np.linalg.norm(x)
            else:
                newImg = torch.tensor(np.reshape(xNew, imShape)).float()
                d = a

            outputs = net(newImg)
            _, predicted = torch.max(outputs.data, 1)

            if predicted == cl:
                return newImg, d

        return newImg, 1


def any(net, img, attack, clamp=False):
    """
    Scale an untargeted adversarial attack perturbation such that it is
    the smallest perturbation in that direction which leads to an
    altered classification.

    Args:
    - net: The neural network model.
    - img: The input image.
    - attack: The perturbation to be scaled.
    - clamp (bool): Whether to clamp the pixel values of the perturbed image.
                    This will keep the values in the interval [0,1].

    Returns:
    - The perturbed image and the relative normwise perturbation achieved.
    """
    outputs = net(img)
    _, cl = torch.max(outputs.data, 1)

    with torch.no_grad():
        imgLen = img.nelement()
        imShape = img.size()
        x = np.reshape(img.flatten().numpy(), (imgLen, 1))
        a = 0.5
        ratio = np.linalg.norm(x) / np.linalg.norm(attack)

        for i in range(1000):
            a = i / 1000
            xNew = x + a * attack * ratio

            if clamp:
                newImg = torch.clamp(torch.tensor(
                    np.reshape(xNew, imShape)), 0, 1).float()
                d = np.linalg.norm(newImg - img) / np.linalg.norm(x)
            else:
                newImg = torch.tensor(np.reshape(xNew, imShape)).float()
                d = a

            outputs = net(newImg)
            _, predicted = torch.max(outputs.data, 1)

            if predicted != cl:
                return newImg, d

        return newImg, 1
