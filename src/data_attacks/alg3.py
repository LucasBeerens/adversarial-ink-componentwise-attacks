import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
from utils import jacobian
import torchvision.transforms as transforms


class Al0():
    def __init__(self, type=0, n=30, a=0.1, disp=False,
                 num=False, tolerance=0, name=None):
        """
        Initialize the alg3.Al0 class. This includes variations of algorithms 3 and 4 from the 
        paper, but without pruning the results and without the pixel value bound constraint. 
        Algorithm 3 without pruning and without pixel value constraint is type 0 tolerance 0. 
        Algorithm 4 without pruning and without pixel value constraint is type 2 tolerance 0.

        Args:
        - type (int): Type of the algorithm changes optimization function. There are four types:
            * 0: ||v||_inf
            * 1: ||v||_inf + ||dx||_2 / ||x||_2
            * 2: ||dx||_2 / ||x||_2
            * 3: ||v||_2
        - n (int): Number of iterations for iterative optimization.
        - a (float): Step size for image perturbation iterations.
        - disp (bool): Whether to return optimization progress instead of final perturbation.
        - num (bool): Whether to use numerical approximation for Jacobian.
        - tolerance (int): Type of tolerance matrix to use. There are three tolerances:
            * 0: image tolerance leading to relative perturbations
            * 1: identity tolerance leading to perturbations without taking into account the image
            * 2: blurred image tolerance leading to relative perturbations but allowing e.g. more
                 changes around the edges of numbers.
        - name: Name parameter.
        """
        self.n = n
        self.a = a
        self.disp = disp
        self.type = type
        self.num = num
        self.tolerance = tolerance
        self.name = name

    def __call__(self, net, img, cl):
        """
        Perform the alg3.Al0 attack.

        Args:
        - net: The neural network model.
        - img: The input image.
        - cl: The target class we wish the perturbed image to be classified as.

        Returns:
        - The result of the Al0 optimization.
        """
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            imVec = np.reshape(newImg, (imgLen, 1))

            # Define the tolerance matrix based on the given type
            if self.tolerance == 0:
                tol = np.diagflat(np.abs(imVec))
            elif self.tolerance == 1:
                tol = torch.tensor(np.eye(imgLen)).float()
            elif self.tolerance == 2:
                blurrer = transforms.GaussianBlur(kernel_size=3, sigma=1)
                tolIm = blurrer(img)
                tol = np.diagflat(np.abs(tolIm))

            v = np.zeros((imgLen, 1))
            dx = np.zeros((imgLen, 1))

            # Define the matrices and vectors based on the given type
            if self.type == 0:
                c1 = np.hstack(
                    [np.ones((1, 1)), np.zeros((1, classCount + imgLen))])
                k1 = np.zeros(1)
            elif self.type == 1:
                c11 = np.hstack([np.ones((1, 1)),
                                 np.zeros((1, classCount + imgLen))])
                c12 = np.hstack([np.zeros((imgLen, classCount+1)),
                                 tol / np.linalg.norm(imVec)])
                c1 = np.vstack([c11, c12])
                k1 = np.vstack([np.zeros(1), -tol@v/np.linalg.norm(imVec)])
            elif self.type == 2:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)),
                                tol / np.linalg.norm(imVec)])
                k1 = -tol@v/np.linalg.norm(imVec)
            elif self.type == 3:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)),
                               np.eye(imgLen) / np.linalg.norm(imVec)])
                k1 = -v/np.linalg.norm(imVec)

            # Compute other matrices defining the optimization problem
            G = np.array([[int(j == cl) for j in range(classCount)] for _ in range(classCount)])
            c2 = np.hstack([np.zeros((classCount, 1)), np.eye(
                classCount) - G, np.zeros((classCount, imgLen))])
            c21 = np.hstack(
                [-np.ones((imgLen, 1)), np.zeros((imgLen, classCount)), np.eye(imgLen)])
            c22 = np.hstack(
                [-np.ones((imgLen, 1)), np.zeros((imgLen, classCount)), -np.eye(imgLen)])
            c2 = np.vstack([c21, c22, c2])
            k2 = cp.Parameter((2*imgLen+classCount, 1))

            c3 = cp.Parameter((classCount, imgLen+classCount+1))
            k3 = cp.Parameter((classCount, 1))

            z = cp.Variable((1 + imgLen + classCount, 1))
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
                k2.value = np.vstack([-v, v, np.zeros((classCount, 1))])
                k3.value = output
                c3.value = np.hstack(
                    [np.zeros((classCount, 1)), np.eye(classCount), -jac@tol])

                # Solve minimisation problem
                prob.solve(solver=cp.ECOS)
                val = z.value

                # Update based on solution to optimisation problem
                dv = val[1+classCount:]
                v += a * dv
                dx = (tol @ v).numpy()
                newImVec = np.reshape(img, (imgLen, 1)) + dx
                newImg = np.reshape(newImVec, img.shape).float()
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
        return scale.specific(net, img, dx, cl, 20)


class Al1():
    def __init__(self, type=0, n=30, a=0.1, disp=False,
                 num=False, tolerance=0, name=None):
        """
        Initialize the alg3.Al1 class. This includes algorithms 3 and 4 from the paper. 
        Algorithm 3 is type 0 tolerance 1. 
        Algorithm 4 is type 2 tolerance 1.

        Args:
        - type (int): Type of the algorithm changes optimization function. There are four types:
            * 0: ||v||_inf
            * 1: ||v||_inf + ||dx||_2 / ||x||_2
            * 2: ||dx||_2 / ||x||_2
            * 3: ||v||_2
        - n (int): Number of iterations for iterative optimization.
        - a (float): Step size for image perturbation iterations.
        - disp (bool): Whether to return optimization progress instead of final perturbation.
        - num (bool): Whether to use numerical approximation for Jacobian.
        - tolerance (int): Type of tolerance matrix to use. There are three tolerances:
            * 0: image tolerance leading to relative perturbations
            * 1: identity tolerance leading to perturbations without taking into account the image
            * 2: blurred image tolerance leading to relative perturbations but allowing e.g. more
                 changes around the edges of numbers.
        - name: Name parameter.
        """
        self.n = n
        self.a = a
        self.disp = disp
        self.type = type
        self.num = num
        self.tolerance = tolerance
        self.name = name

    def __call__(self, net, img, cl):
        """
        Perform the alg3.Al1 attack.

        Args:
        - net: The neural network model.
        - img: The input image.
        - cl: The target class we wish the perturbed image to be classified as.

        Returns:
        - The result of the Al0 optimization.
        """
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            imVec = np.reshape(newImg, (imgLen, 1)).numpy()

            # Define the tolerance matrix based on the given type
            if self.tolerance == 0:
                tol = np.diagflat(np.abs(imVec))
            elif self.tolerance == 1:
                tol = np.eye(imgLen)
            elif self.tolerance == 2:
                blurrer = transforms.GaussianBlur(kernel_size=3, sigma=1)
                tolIm = np.array(blurrer(img))
                tol = np.diagflat(np.abs(tolIm))

            v = np.zeros((imgLen, 1))
            dx = np.zeros((imgLen, 1))

            # Define the matrices and vectors based on the given type
            if self.type == 0:
                c1 = np.hstack(
                    [np.ones((1, 1)), np.zeros((1, classCount + imgLen))])
                k1 = np.zeros(1)
            elif self.type == 1:
                c11 = np.hstack(
                    [np.ones((1, 1)), np.zeros((1, classCount + imgLen))])
                c12 = np.hstack(
                    [np.zeros((imgLen, classCount+1)), tol / np.linalg.norm(imVec)])
                c1 = np.vstack([c11, c12])
                k1 = np.vstack([np.zeros(1), -tol@v/np.linalg.norm(imVec)])
            elif self.type == 2:
                c1 = np.hstack(
                    [np.zeros((imgLen, classCount+1)), tol / np.linalg.norm(imVec)])
                k1 = -tol@v/np.linalg.norm(imVec)
            elif self.type == 3:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)),
                               np.eye(imgLen) / np.linalg.norm(imVec)])
                k1 = -v/np.linalg.norm(imVec)

            # Compute other matrices defining the optimization problem
            G = np.array([[int(j == cl) for j in range(classCount)]
                         for _ in range(classCount)])
            c2 = np.hstack([np.zeros((classCount, 1)), np.eye(
                classCount) - G, np.zeros((classCount, imgLen))])
            c21 = np.hstack(
                [-np.ones((imgLen, 1)), np.zeros((imgLen, classCount)), np.eye(imgLen)])
            c22 = np.hstack(
                [-np.ones((imgLen, 1)), np.zeros((imgLen, classCount)), -np.eye(imgLen)])
            c23 = np.hstack([np.zeros((imgLen, classCount+1)), a*tol])
            c24 = np.hstack([np.zeros((imgLen, classCount+1)), -a*tol])
            c2 = np.vstack([c21, c22, c2, c23, c24])
            k2 = cp.Parameter((4*imgLen+classCount, 1))

            c3 = cp.Parameter((classCount, imgLen+classCount+1))
            k3 = cp.Parameter((classCount, 1))

            z = cp.Variable((1 + imgLen + classCount, 1))
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
                k2.value = np.vstack([-v, v, np.zeros((classCount, 1)), 
                                      np.ones((imgLen, 1))-imVec-tol@v, 
                                      imVec+tol@v])
                k3.value = output
                c3.value = np.hstack(
                    [np.zeros((classCount, 1)), np.eye(classCount), -jac@tol])

                # Solve minimisation problem
                prob.solve(solver=cp.ECOS)
                val = z.value
                if val is None:
                    return img, 1

                # Update based on solution to optimisation problem
                dv = val[1+classCount:]
                v += a * dv
                dx = tol @ v
                newImVec = np.reshape(img, (imgLen, 1)) + dx
                newImg = np.reshape(newImVec, img.shape).float()
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
