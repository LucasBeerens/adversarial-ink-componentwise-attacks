import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
import matplotlib.pyplot as plt
from utils import jacobian
import torchvision.transforms as transforms

class Al0():
    def __init__(self,type=0,n=30,a=0.1,disp=False,num=False,tolerance=0,name=None):
        self.n = n
        self.a = a
        self.disp = disp
        self.type = type
        self.num = num
        self.tolerance=tolerance
        self.name = name

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
            if self.tolerance==0:
                tol = np.diagflat(np.abs(imVec))
            elif self.tolerance==1:
                tol = torch.tensor(np.eye(imgLen)).float()
            elif self.tolerance==2:
                blurrer = transforms.GaussianBlur(kernel_size=3,sigma=1)
                tolIm = blurrer(img)
                tol = np.diagflat(np.abs(tolIm))

            v = np.zeros((imgLen,1))
            dx = np.zeros((imgLen,1))

            if self.type == 0:
                c1 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                k1 = np.zeros(1)
            elif self.type == 1:
                c11 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                c12 = np.hstack([np.zeros((imgLen, classCount+1)), tol / np.linalg.norm(imVec)])
                c1 = np.vstack([c11,c12])
                k1 = np.vstack([np.zeros(1),-tol@v/np.linalg.norm(imVec)])
            elif self.type == 2:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)), tol / np.linalg.norm(imVec)])
                k1 = -tol@v/np.linalg.norm(imVec)
            elif self.type == 3:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)), np.eye(imgLen) / np.linalg.norm(imVec)])
                k1 = -v/np.linalg.norm(imVec)
            
            
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
                if not self.num:
                    jac = torch.autograd.functional.jacobian(net,newImg)
                    jac = jac.view(classCount, imgLen)
                else:
                    jac = jacobian.approx(net,newImg)

                k2.value = np.vstack([-v,v,np.zeros((classCount,1))])
                k3.value = output
                c3.value = np.hstack([np.zeros((classCount,1)),np.eye(classCount),-jac@tol])

                prob.solve(solver=cp.ECOS)
                
                val = z.value

                dv = val[1+classCount:]
                v += a * dv
                dx = (tol @ v).numpy()
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
                return epss
        return scale.specific(net,img,dx,cl,20)

class Al1():
    def __init__(self,type=0,n=30,a=0.1,disp=False,num=False,tolerance=0,name=None):
        self.n = n
        self.a = a
        self.disp = disp
        self.type = type
        self.num = num
        self.tolerance=tolerance
        self.name = name

    def __call__(self,net,img,cl,style='-',lw=3):
        n = self.n
        a = self.a
        with torch.no_grad():
            net.eval()
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()
            newImg = img
            imVec = np.reshape(newImg,(imgLen,1)).numpy()
            if self.tolerance==0:
                tol = np.diagflat(np.abs(imVec))
            elif self.tolerance==1:
                tol = np.eye(imgLen)
            elif self.tolerance==2:
                blurrer = transforms.GaussianBlur(kernel_size=3,sigma=1)
                tolIm = np.array(blurrer(img))
                tol = np.diagflat(np.abs(tolIm))

            v = np.zeros((imgLen,1))
            dx = np.zeros((imgLen,1))

            if self.type == 0:
                c1 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                k1 = np.zeros(1)
            elif self.type == 1:
                c11 = np.hstack([np.ones((1,1)),np.zeros((1,classCount +  imgLen))])
                c12 = np.hstack([np.zeros((imgLen, classCount+1)), tol / np.linalg.norm(imVec)])
                c1 = np.vstack([c11,c12])
                k1 = np.vstack([np.zeros(1),-tol@v/np.linalg.norm(imVec)])
            elif self.type == 2:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)), tol / np.linalg.norm(imVec)])
                k1 = -tol@v/np.linalg.norm(imVec)
            elif self.type == 3:
                c1 = np.hstack([np.zeros((imgLen, classCount+1)), np.eye(imgLen) / np.linalg.norm(imVec)])
                k1 = -v/np.linalg.norm(imVec)
            
            G = np.array([[int(j==cl) for j in range(classCount)] for _ in range(classCount)])
            c2 = np.hstack([np.zeros((classCount,1)),np.eye(classCount) - G,np.zeros((classCount,imgLen))])
            c21 = np.hstack([-np.ones((imgLen,1)),np.zeros((imgLen,classCount)),np.eye(imgLen)])
            c22 = np.hstack([-np.ones((imgLen,1)),np.zeros((imgLen,classCount)),-np.eye(imgLen)])
            c23 = np.hstack([np.zeros((imgLen,classCount+1)),a*tol])
            c24 = np.hstack([np.zeros((imgLen,classCount+1)),-a*tol])
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
                if not self.num:
                    jac = torch.autograd.functional.jacobian(net,newImg)
                    jac = jac.view(classCount, imgLen)
                else:
                    jac = jacobian.approx(net,newImg)

                k2.value = np.vstack([-v,v,np.zeros((classCount,1)),np.ones((imgLen,1))-imVec-tol@v,imVec+tol@v])
                k3.value = output
                c3.value = np.hstack([np.zeros((classCount,1)),np.eye(classCount),-jac@tol])

                prob.solve(solver=cp.SCS,ignore_dpp = True)
                
                val = z.value
                if val is None:
                    return img, 1

                dv = val[1+classCount:]
                v += a * dv
                dx = tol @ v
                newImVec = np.reshape(img,(imgLen,1)) + dx
                newImg = np.reshape(newImVec,img.shape).float()
                if self.disp:
                    _, epss[i] = scale.specific(net,img,dx,cl,20,True)
                
                output = net(newImg)
                _, predicted = torch.max(output.data,1)
                if predicted == cl:
                    break
                output = output.numpy().transpose()
            if self.disp:
                return epss
        return scale.specific(net,img,dx,cl,20,clamp=True)