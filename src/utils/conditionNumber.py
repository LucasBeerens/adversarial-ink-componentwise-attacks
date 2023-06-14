import torch
import numpy as np


def normwise(net, img):
    """
    Calculate the normwise condition number of the neural network.

    Args:
    - net: The neural network model.
    - img: The input image.

    Returns:
    - The normwise condition number
    """
    with torch.no_grad():
        output = net(img).numpy().transpose()
        classCount = output.size
        imgLen = img.nelement()

        # Compute the Jacobian of the network with respect to the input image
        jac = torch.autograd.functional.jacobian(net, img)
        jac = jac.view(classCount, imgLen)

        # Calculate the norms of the output, Jacobian, and input image
        outputNorm = np.linalg.norm(output)
        jacNorm = np.linalg.norm(jac, ord=2)
        imNorm = np.linalg.norm(img)

        # Calculate the normwise condition number
        return jacNorm * imNorm / outputNorm


def componentwise(net, img, tol):
    """
    Calculate the componentwise sensitivity of the neural network.

    Args:
    - net: The neural network model.
    - img: The input image.
    - tol: The tolerance vector.

    Returns:
    - The componentwise condition number
    """
    with torch.no_grad():
        output = net(img).numpy().transpose()
        classCount = output.size
        imgLen = img.nelement()

        # Compute the Jacobian of the network with respect to the input image
        jac = torch.autograd.functional.jacobian(net, img)
        jac = jac.view(classCount, imgLen)

        # Calculate the componentwise condition number
        B = np.abs(jac) @ np.abs(tol)
        output = net(img).numpy()
        return np.linalg.norm(B, ord=np.inf) / np.linalg.norm(output, ord=np.inf)
