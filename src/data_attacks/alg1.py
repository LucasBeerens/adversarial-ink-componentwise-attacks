import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
from utils import jacobian


class Al0:
    def __init__(self, num=False) -> None:
        """
        Initialize alg1.Al0 class. This corresponds to an alternative version
        of Algorithm 1 from the paper without the pruning.

        Args:
        - num (bool): Whether to use numerical approximation for Jacobian.
        """
        self.num = num

    def __call__(self, net, img, cl):
        """
        Perform alg1.AL0 attack.

        Args:
        - net: The neural network model.
        - img: The input image.
        - cl: The target class we wish the perturbed image to be classified as.

        Returns:
        - The result of the alg1.AL0 optimization.
        """
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()

            # Compute the Jacobian and its pseudo-inverse
            if not self.num:
                jac = torch.autograd.functional.jacobian(net, img)
                jac = jac.view(classCount, imgLen)
            else:
                jac = jacobian.approx(net, img)
            pi = jac.pinverse().numpy()

            jac = jac.numpy()

        # Compute matrices and vectors that represent the optimization problem
        c1 = pi
        k1 = pi @ output

        G = np.array([[int(j == cl) for j in range(classCount)] for _ in range(classCount)])
        c2 = np.eye(classCount) - G
        k2 = np.zeros((classCount, 1))

        # Setup and solve minimization problem
        y = cp.Variable((classCount, 1))
        obj = cp.Minimize(cp.sum_squares(c1 @ y - k1))
        eqCons = [c2 @ y <= k2]
        prob = cp.Problem(obj, eqCons)
        prob.solve()
        val = y.value

        # Compute the corresponding image perturbation
        dy = val - output
        dx = pi @ dy

        # Return the scaled version of the perturbed image and its epsilon
        return scale.specific(net, img, dx, cl)


class Al1:
    def __init__(self, num=False) -> None:
        """
        Initialize alg1.Al1 class. This corresponds Algorithm 1 from the paper.

        Args:
        - num (bool): Whether to use numerical approximation for Jacobian.
        """
        self.num = num

    def __call__(self, net, img, cl):
        """
        Perform alg1.Al1 attack.

        Args:
        - net: The neural network model.
        - img: The input image.
        - cl: The target class we wish the perturbed image to be classified as.

        Returns:
        - The result of the alg1.AL1 optimization.
        """
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()

            # Compute the Jacobian and its pseudo-inverse
            if not self.num:
                jac = torch.autograd.functional.jacobian(net, img)
                jac = jac.view(classCount, imgLen)
            else:
                jac = jacobian.approx(net, img)
            pi = jac.pinverse().numpy()

            jac = jac.numpy()

        # Compute matrices and vectors that represent the optimization problem
        c1 = pi
        k1 = pi @ output

        G = np.array([[int(j == cl) for j in range(classCount)] for _ in range(classCount)])
        c2 = np.eye(classCount) - G
        k2 = np.zeros((classCount, 1))

        # Setup and solve minimization problem
        y = cp.Variable((classCount, 1))
        obj = cp.Minimize(cp.sum_squares(c1 @ y - k1))
        eqCons = [c2 @ y <= k2]
        prob = cp.Problem(obj, eqCons)
        prob.solve()
        val = y.value

        # Compute the corresponding image perturbation
        dy = val - output
        dx = pi @ dy

        # Return the scaled and pruned version of the perturbed image and its epsilon
        return scale.specific(net, img, dx, cl, 20, True)
