import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
import matplotlib.pyplot as plt

class Al0():
    def __init__(self,type=0,n=150,a=0.01,disp=False):
        self.n = n
        self.a = a
        self.disp = disp
        self.type = type

    def __call__(self,net,img,cl):
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            imVec = np.reshape(newImg,(imgLen,1))
            diagIm = np.diagflat(np.abs(imVec))

            v = np.zeros((imgLen,1))
            dx = np.zeros((imgLen,1))

            if self.type == 0:
                c1 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                k1 = np.zeros(1)
            elif self.type == 1:
                c11 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                c12 = np.hstack([np.zeros((imgLen, classCount+1)), diagIm / np.linalg.norm(imVec)])
                c1 = np.vstack([c11,c12])
                k1 = np.vstack([np.zeros(1),-diagIm@v/np.linalg.norm(imVec)])
            elif self.type == 2:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)), diagIm / np.linalg.norm(imVec)])
                k1 = -diagIm@v/np.linalg.norm(imVec)
            
            
            G = np.array([[int(j==cl) for j in range(classCount)] for _ in range(classCount)])
            c2 = np.hstack([np.zeros((classCount,1)),np.eye(classCount) - G,np.zeros((classCount,imgLen))])
            c21 = np.hstack([-np.ones((imgLen,1)),np.zeros((imgLen,classCount)),np.eye(imgLen)])
            c22 = np.hstack([-np.ones((imgLen,1)),np.zeros((imgLen,classCount)),-np.eye(imgLen)])
            c2 = np.vstack([c21,c22,c2])
            k2 = cp.Parameter((2*imgLen+classCount,1))
            
            c3 = cp.Parameter((classCount,imgLen+classCount+1))
            k3 = cp.Parameter((classCount,1))
            
            z = cp.Variable((1 + imgLen + classCount,1))
            obj = cp.Minimize(cp.sum_squares(c1 @ z - k1))
            eqCons = [c2 @ z <= k2, c3 @ z == k3]
            prob = cp.Problem(obj, eqCons)
            assert prob.is_dcp(dpp=True)

            epss = np.zeros(n)

            for i in range(n):
                jac = torch.autograd.functional.jacobian(net,newImg)
                jac = jac.view(classCount, imgLen)

                k2.value = np.vstack([-v,v,np.zeros((classCount,1))])
                k3.value = output
                c3.value = np.hstack([np.zeros((classCount,1)),np.eye(classCount),-jac@diagIm])

                prob.solve(solver=cp.ECOS)
                
                val = z.value

                dv = val[1+classCount:]
                v += a * dv
                dx = (diagIm @ v).numpy()
                newImVec = np.reshape(img,(imgLen,1)) + dx
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
    def __init__(self,type=0,n=150,a=0.01,disp=False):
        self.n = n
        self.a = a
        self.disp = disp
        self.type = type

    def __call__(self,net,img,cl):
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            imVec = np.reshape(newImg,(imgLen,1)).numpy()
            diagIm = np.diagflat(np.abs(imVec))

            v = np.zeros((imgLen,1))
            dx = np.zeros((imgLen,1))

            if self.type == 0:
                c1 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                k1 = np.zeros(1)
            elif self.type == 1:
                c11 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                c12 = np.hstack([np.zeros((imgLen, classCount+1)), diagIm / np.linalg.norm(imVec)])
                c1 = np.vstack([c11,c12])
                k1 = np.vstack([np.zeros(1),-diagIm@v/np.linalg.norm(imVec)])
            elif self.type == 2:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)), diagIm / np.linalg.norm(imVec)])
                k1 = -diagIm@v/np.linalg.norm(imVec)
            
            G = np.array([[int(j==cl) for j in range(classCount)] for _ in range(classCount)])
            c2 = np.hstack([np.zeros((classCount,1)),np.eye(classCount) - G,np.zeros((classCount,imgLen))])
            c21 = np.hstack([-np.ones((imgLen,1)),np.zeros((imgLen,classCount)),np.eye(imgLen)])
            c22 = np.hstack([-np.ones((imgLen,1)),np.zeros((imgLen,classCount)),-np.eye(imgLen)])
            c23 = np.hstack([np.zeros((imgLen,classCount+1)),a*diagIm])
            c24 = np.hstack([np.zeros((imgLen,classCount+1)),-a*diagIm])
            c2 = np.vstack([c21,c22,c2,c23,c24])
            k2 = cp.Parameter((4*imgLen+classCount,1))
            
            c3 = cp.Parameter((classCount,imgLen+classCount+1))
            k3 = cp.Parameter((classCount,1))
            
            z = cp.Variable((1 + imgLen + classCount,1))
            obj = cp.Minimize(cp.sum_squares(c1 @ z - k1))
            eqCons = [c2 @ z <= k2, c3 @ z == k3]
            prob = cp.Problem(obj, eqCons)
            assert prob.is_dcp(dpp=True)

            epss = np.zeros(n)

            for i in range(n):
                jac = torch.autograd.functional.jacobian(net,newImg)
                jac = jac.view(classCount, imgLen)

                k2.value = np.vstack([-v,v,np.zeros((classCount,1)),np.ones((imgLen,1))-imVec-diagIm@v,imVec+diagIm@v])
                k3.value = output
                c3.value = np.hstack([np.zeros((classCount,1)),np.eye(classCount),-jac@diagIm])

                prob.solve(solver=cp.ECOS,ignore_dpp = True)
                
                val = z.value
                dv = val[1+classCount:]
                v += a * dv
                dx = diagIm @ v
                newImVec = np.reshape(img,(imgLen,1)) + dx
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