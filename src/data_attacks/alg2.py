import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
import matplotlib.pyplot as plt
from utils import jacobian

class Al0():
    def __init__(self,n=150,a=0.01,disp=False,num=False):
        self.n = n
        self.a = a
        self.disp = disp
        self.num = num

    def __call__(self,net,img,cl):
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            dx = np.zeros((imgLen,1))

            c1 = np.hstack([np.eye(imgLen),np.zeros((imgLen,classCount))])
            k1 = cp.Parameter((imgLen,1))
            
            G = np.array([[int(j==cl) for j in range(classCount)] for _ in range(classCount)])
            c2 = np.hstack([np.zeros((classCount,imgLen)),np.eye(classCount) - G])
            k2 = np.zeros((classCount,1))
            
            c3 = cp.Parameter((classCount,imgLen+classCount))
            k3 = cp.Parameter((classCount,1))
            
            z = cp.Variable((imgLen + classCount,1))
            obj = cp.Minimize(cp.sum_squares(c1 @ z - k1))
            eqCons = [c2 @ z <= k2, c3 @ z == k3]
            prob = cp.Problem(obj, eqCons)
            assert prob.is_dcp(dpp=True)

            epss = np.zeros(n)

            for i in range(n):
                if not self.num:
                    jac = torch.autograd.functional.jacobian(net,newImg)
                    jac = jac.view(classCount, imgLen)
                else:
                    jac = jacobian.approx(net,newImg)

                k1.value = -dx
                k3.value = output
                c3.value = np.hstack([-jac, np.eye(classCount)])

                prob.solve()
                
                val = z.value

                delx = val[:imgLen]
                dx += a * delx

                imVec = np.reshape(newImg,(imgLen,1))
                newImVec = imVec + a * delx
                newImg = np.reshape(newImVec,img.shape).float()
                if self.disp:
                    _, epss[i] = scale.specific(net,img,dx,cl,20)
                
                output = net(newImg)
                _, predicted = torch.max(output.data,1)
                if predicted == cl:
                    break
                output = output.numpy().transpose()
            if self.disp:
                plt.plot(epss)
                plt.show()
        return scale.specific(net,img,dx,cl,20)


class Al1():
    def __init__(self,n=150,a=0.01,disp=False,num=False):
        self.n = n
        self.a = a
        self.disp = disp
        self.num = num

    def __call__(self,net,img,cl):
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            newImVec = np.reshape(newImg,(imgLen,1))
            dx = np.zeros((imgLen,1))

            c1 = np.hstack([np.eye(imgLen),np.zeros((imgLen,classCount))])
            k1 = cp.Parameter((imgLen,1))
            
            G = np.array([[int(j==cl) for j in range(classCount)] for _ in range(classCount)])
            c2 = np.hstack([np.zeros((classCount,imgLen)),np.eye(classCount) - G])
            c21 = np.hstack([np.eye(imgLen), np.zeros((imgLen,classCount))])
            c22 = np.hstack([-np.eye(imgLen), np.zeros((imgLen,classCount))])
            c2 = np.vstack([c2,c21,c22])
            k2 = cp.Parameter((2*imgLen + classCount,1))
            
            c3 = cp.Parameter((classCount,imgLen+classCount))
            k3 = cp.Parameter((classCount,1))
            
            z = cp.Variable((imgLen + classCount,1))
            obj = cp.Minimize(cp.sum_squares(c1 @ z - k1))
            eqCons = [c2 @ z <= k2, c3 @ z == k3]
            prob = cp.Problem(obj, eqCons)
            assert prob.is_dcp(dpp=True)

            epss = np.zeros(n)

            for i in range(n):
                if not self.num:
                    jac = torch.autograd.functional.jacobian(net,newImg)
                    jac = jac.view(classCount, imgLen)
                else:
                    jac = jacobian.approx(net,newImg)

                k1.value = -dx
                k2.value = np.vstack([np.zeros((classCount,1)),(np.ones((imgLen,1)) - newImVec.numpy()),newImVec.numpy()])
                k3.value = output
                c3.value = np.hstack([-jac, np.eye(classCount)])

                prob.solve(ignore_dpp = True)
                
                val = z.value

                delx = val[:imgLen]
                dx += a * delx

                imVec = np.reshape(newImg,(imgLen,1))
                newImVec = imVec + a * delx
                newImg = np.reshape(newImVec,img.shape).float()
                if self.disp:
                    _, epss[i] = scale.specific(net,img,dx,cl,20)
                
                output = net(newImg)
                _, predicted = torch.max(output.data,1)
                if predicted == cl:
                    break
                output = output.numpy().transpose()
            if self.disp:
                plt.plot(epss)
                plt.show()
        return scale.specific(net,img,dx,cl,20,clamp=True)