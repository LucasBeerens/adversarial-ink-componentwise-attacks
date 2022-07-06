from tkinter.tix import Tree
import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale

def attack(net,img,cl):
    with torch.no_grad():
        net.eval()
        output = net(img).numpy().transpose()
        classCount = output.size
        imgLen = img.nelement()

        jac = torch.autograd.functional.jacobian(net,img)
        jac = jac.view(classCount, imgLen)
        pi = jac.pinverse().numpy()

        jac = jac.numpy()

    c1 = pi
    k1 = pi @ output

    G = np.array([[int(j==cl) for j in range(classCount)] for _ in range(classCount)])
    c2 = np.eye(classCount) - G
    k2 = np.zeros((classCount,1))

    y = cp.Variable((classCount,1))
    obj = cp.Minimize(cp.sum_squares(c1 @ y - k1))
    eqCons = [c2 @ y <= k2]
    prob = cp.Problem(obj, eqCons)
    prob.solve()
    
    val = y.value

    dy = val - output
    dx = pi @ dy

    return scale.scale(net,img,dx,cl,20)