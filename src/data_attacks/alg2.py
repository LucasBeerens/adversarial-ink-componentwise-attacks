import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
from utils import jacobian


class Al0():
    def __init__(self, n=30, a=0.1, disp=False, num=False, name=None):
        """
        Initialize the alg2.Al0 class. This corresponds to an alternative version
        of Algorithm 2 from the paper without the pixel value bound constraint.

        Args:
        - n (int): Number of iterations for iterative optimization.
        - a (float): Step size for image perturbation iterations.
        - disp (bool): Whether to return optimization progress instead of final perturbation.
        - num (bool): Whether to use numerical approximation for Jacobian.
        - name: Name parameter.
        """
        self.n = n
        self.a = a
        self.disp = disp
        self.num = num
        self.name = name

    def __call__(self, net, img, cl):
        """
        Perform alg2.AL1 attack.

        Args:
        - net: The neural network model.
        - img: The input image.
        - cl: The target class we wish the perturbed image to be classified as.

        Returns:
        - The result of the alg2.AL0 optimization.
        """
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            dx = np.zeros((imgLen, 1))

            # Compute/initialize matrices and vectors that represent the optimization problem
            c1 = np.hstack([np.eye(imgLen), np.zeros((imgLen, classCount))])
            k1 = cp.Parameter((imgLen, 1))

            G = np.array([[int(j == cl) for j in range(classCount)]
                         for _ in range(classCount)])
            c2 = np.hstack(
                [np.zeros((classCount, imgLen)), np.eye(classCount) - G])
            k2 = np.zeros((classCount, 1))

            c3 = cp.Parameter((classCount, imgLen+classCount))
            k3 = cp.Parameter((classCount, 1))

            # Setup minimization problem.
            z = cp.Variable((imgLen + classCount, 1))
            obj = cp.Minimize(cp.sum_squares(c1 @ z - k1))
            eqCons = [c2 @ z <= k2, c3 @ z == k3]
            prob = cp.Problem(obj, eqCons)
            assert prob.is_dcp(dpp=True)

            epss = np.zeros(n)

            for i in range(n):
                # Compute the Jacobian and its pseudo-inverse
                if not self.num:
                    jac = torch.autograd.functional.jacobian(net, newImg)
                    jac = jac.view(classCount, imgLen)
                else:
                    jac = jacobian.approx(net, newImg)

                # Compute optimization matrices and vectors that update each iteration
                k1.value = -dx
                k3.value = output
                c3.value = np.hstack([-jac, np.eye(classCount)])

                # Solve minimisation problem
                prob.solve()
                val = z.value

                # Update based on solution to optimisation problem
                delx = val[:imgLen]
                dx += a * delx
                imVec = np.reshape(newImg, (imgLen, 1))
                newImVec = imVec + a * delx
                newImg = np.reshape(newImVec, img.shape).float()

                # If we just want the progression through iterations, compute that
                if self.disp:
                    _, epss[i] = scale.specific(net, img, dx, cl, 20)

                # Compute outputs and stop iteration if we have arrived
                output = net(newImg)
                _, predicted = torch.max(output.data, 1)
                if predicted == cl:
                    break
                output = output.numpy().transpose()
            if self.disp:
                return epss

        # Return the scaled and pruned version of the perturbed image and its epsilon
        return scale.specific(net, img, dx, cl, 20, clamp=True)


class Al1():
    def __init__(self, n=30, a=0.1, disp=False, num=False, name=None):
        """
        Initialize the alg2.Al0 class. This corresponds Algorithm 2 from the paper.

        Args:
        - n (int): Number of iterations for iterative optimization.
        - a (float): Step size for image perturbation iterations.
        - disp (bool): Whether to return optimization progress instead of final perturbation.
        - num (bool): Whether to use numerical approximation for Jacobian.
        - name: Name parameter.
        """
        self.n = n
        self.a = a
        self.disp = disp
        self.num = num
        self.name = name

    def __call__(self, net, img, cl):
        """
        Perform alg2.AL0 attack.

        Args:
        - net: The neural network model.
        - img: The input image.
        - cl: The target class we wish the perturbed image to be classified as.

        Returns:
        - The result of the alg2.AL0 optimization.
        """
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            newImVec = np.reshape(newImg, (imgLen, 1))
            dx = np.zeros((imgLen, 1))

            # Compute/initialize matrices and vectors that represent the optimization problem
            c1 = np.hstack([np.eye(imgLen), np.zeros((imgLen, classCount))])
            k1 = cp.Parameter((imgLen, 1))

            G = np.array([[int(j == cl) for j in range(classCount)]
                         for _ in range(classCount)])
            c2 = np.hstack(
                [np.zeros((classCount, imgLen)), np.eye(classCount) - G])
            c21 = np.hstack([np.eye(imgLen), np.zeros((imgLen, classCount))])
            c22 = np.hstack([-np.eye(imgLen), np.zeros((imgLen, classCount))])
            c2 = np.vstack([c2, c21, c22])
            k2 = cp.Parameter((2*imgLen + classCount, 1))

            c3 = cp.Parameter((classCount, imgLen+classCount))
            k3 = cp.Parameter((classCount, 1))

            # Setup minimization problem.
            z = cp.Variable((imgLen + classCount, 1))
            obj = cp.Minimize(cp.sum_squares(c1 @ z - k1))
            eqCons = [c2 @ z <= k2, c3 @ z == k3]
            prob = cp.Problem(obj, eqCons)
            assert prob.is_dcp(dpp=True)

            epss = np.zeros(n)

            for i in range(n):
                # Compute the Jacobian and its pseudo-inverse
                if not self.num:
                    jac = torch.autograd.functional.jacobian(net, newImg)
                    jac = jac.view(classCount, imgLen)
                else:
                    jac = jacobian.approx(net, newImg)

                # Compute optimization matrices and vectors that update each iteration
                k1.value = -dx
                k2.value = np.vstack([np.zeros((classCount, 1)),
                                      (np.ones((imgLen, 1)) - newImVec.numpy()),
                                      newImVec.numpy()])
                k3.value = output
                c3.value = np.hstack([-jac, np.eye(classCount)])

                # Solve minimisation problem
                prob.solve()
                val = z.value

                # Update based on solution to optimisation problem
                delx = val[:imgLen]
                dx += a * delx
                imVec = np.reshape(newImg, (imgLen, 1))
                newImVec = imVec + a * delx
                newImg = np.reshape(newImVec, img.shape).float()

                # If we just want the progression through iterations, compute that
                if self.disp:
                    _, epss[i] = scale.specific(net, img, dx, cl, 20, True)

                # Compute outputs and stop iteration if we have arrived
                output = net(newImg)
                _, predicted = torch.max(output.data, 1)
                if predicted == cl:
                    break
                output = output.numpy().transpose()
            if self.disp:
                return epss

        # Return the scaled and pruned version of the perturbed image and its epsilon
        return scale.specific(net, img, dx, cl, 20, clamp=True)
