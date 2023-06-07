import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
from utils import jacobian


class Al0:
    def __init__(self, num=False) -> None:
        self.num = num

    def __call__(self, net, img, cl):
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()

            if not self.num:
                jac = torch.autograd.functional.jacobian(net, img)
                jac = jac.view(classCount, imgLen)
            else:
                jac = jacobian.approx(net, img)
            pi = jac.pinverse().numpy()

            jac = jac.numpy()

        c1 = pi
        k1 = pi @ output

        G = np.array([[int(j == cl) for j in range(classCount)]
                     for _ in range(classCount)])
        c2 = np.eye(classCount) - G
        k2 = np.zeros((classCount, 1))

        y = cp.Variable((classCount, 1))
        obj = cp.Minimize(cp.sum_squares(c1 @ y - k1))
        eqCons = [c2 @ y <= k2]
        prob = cp.Problem(obj, eqCons)
        prob.solve()
        val = y.value

        dy = val - output
        dx = pi @ dy

        return scale.specific(net, img, dx, cl)


class Al1:
    def __init__(self, num=False) -> None:
        self.num = num

    def __call__(self, net, img, cl):
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()

            if not self.num:
                jac = torch.autograd.functional.jacobian(net, img)
                jac = jac.view(classCount, imgLen)
            else:
                jac = jacobian.approx(net, img)
            pi = jac.pinverse().numpy()

            jac = jac.numpy()

        c1 = pi
        k1 = pi @ output

        G = np.array([[int(j == cl) for j in range(classCount)]
                     for _ in range(classCount)])
        c2 = np.eye(classCount) - G
        k2 = np.zeros((classCount, 1))

        y = cp.Variable((classCount, 1))
        obj = cp.Minimize(cp.sum_squares(c1 @ y - k1))
        eqCons = [c2 @ y <= k2]
        prob = cp.Problem(obj, eqCons)
        prob.solve()

        val = y.value

        dy = val - output
        dx = pi @ dy

        return scale.specific(net, img, dx, cl, 20, True)
