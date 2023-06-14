import torch
import numpy as np


def approx(net, img, eps=1e-3):
    """
    Approximate the Jacobian of the neural network using finite differences.

    Args:
    - net: The neural network model.
    - img: The input image.
    - eps: The perturbation magnitude for finite differences.

    Returns:
    - The approximate Jacobian matrix.
    """
    net.eval()
    output = net(img)
    imShape = img.size()
    imWidth = imShape[-1]
    classCount = output.nelement()
    imgLen = img.nelement()
    unitTensors = torch.zeros((imWidth, imWidth, imWidth, imWidth))
    jac = np.zeros((imgLen, classCount))

    # Iterate over each pixel in the image
    for i in range(imWidth):
        for j in range(imWidth):
            unitTensors[i, j, i, j] = 1

            # Perturb the image by adding a unit tensor multiplied by the perturbation magnitude
            perturbedImg = img + unitTensors[i, j, :, :].view(imShape) * eps

            # Calculate the difference in outputs with respect to the perturbation
            new = (net(perturbedImg) - output) / eps

            # Store the difference in the Jacobian matrix
            jac[i * imWidth + j, :] = new

    # Return the jacobian
    return torch.transpose(torch.tensor(jac), 0, 1).float()
