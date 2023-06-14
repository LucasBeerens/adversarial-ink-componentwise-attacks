import numpy as np
import torch


class Any:
    def __init__(self, alg) -> None:
        """
        Initialize the Any class. This takes an attack algorithm that targets
        a specific class and wraps it in a such a way that it can be used as
        an attack without target class.

        Args:
        - alg: The attack algorithm to use.
        """
        self.alg = alg

    def __call__(self, net, img):
        """
        Apply the attack algorithm to find the best adversarial image.

        Args:
        - net: The neural network model.
        - img: The input image.

        Returns:
        - The best adversarial image and the corresponding epsilon value.
        """
        outputs = net(img)
        _, cl = torch.max(outputs.data, 1)

        outputSize = net(img).size(-1)
        resultEps = np.ones(outputSize-1)
        resultImg = []

        # Iterate over all classes except the correct class
        for c in range(outputSize-1):
            target = c + (c >= cl)
            newImg, eps = self.alg(net, img, target)
            resultEps[c] = eps
            resultImg.append(newImg)

        # Find the best adversarial image among the target classes
        bestEps = np.min(resultEps)
        index = np.argmin(resultEps)
        bestImg = resultImg[index]

        return bestImg, bestEps
