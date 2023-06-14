import torch
from utils import scaleAttack as scale


class Att:
    def __init__(self, alg) -> None:
        """
        Initialize the Att class. This class wraps adversarial
        attack algorithms from the torchattacks library to make
        them compatible with the analysis that is done.


        Args:
        - alg: The attack algorithm to be used.
        """
        self.alg = alg
        pass

    def __call__(self, net, img):
        """
        Perform the attack. The scale function is used at the end
        for fair comparison with the other algorithms.

        Args:
        - net: The neural network model.
        - img: The input image.

        Returns:
        - The perturbed image after the attack and its epsilon
        """
        outputs = net(img)
        _, predicted = torch.max(outputs.data, 1)
        label = torch.tensor([predicted])

        # Generate perturbation
        adv_image = self.alg(img, label)
        delta = adv_image - img
        delta = torch.reshape(delta, (delta.nelement(), 1)).numpy()

        # Return the scaled and pruned version of the perturbed image and its epsilon
        return scale.any(net, img, delta, clamp=True)
