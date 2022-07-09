import numpy as np
import torch

def specific(net, img, attack, cl, n=10, clamp=False):
    with torch.no_grad():
        imgLen = img.nelement()
        imShape = img.size()
        x = np.reshape(img.flatten().numpy(),(imgLen,1))
        a = 0.5
        ratio = np.linalg.norm(x) / np.linalg.norm(attack)
        for i in range(1000):
            a = i/1000
            xNew = x +  a * attack * ratio
            if clamp:
                newImg = torch.clamp(torch.tensor(np.reshape(xNew, imShape)),0,1).float()
                d = np.linalg.norm(newImg -img)/np.linalg.norm(x)
            else:
                newImg = torch.tensor(np.reshape(xNew, imShape)).float()
                d=a
            outputs = net(newImg)
            _, predicted = torch.max(outputs.data,1)
            if predicted == cl:
                return newImg, d
        return newImg, 1

        for i in range(2,n+1):
            xNew = x +  a * attack * ratio
            newImg = torch.tensor(np.reshape(xNew, imShape)).float()
            outputs = net(newImg)
            _, predicted = torch.max(outputs.data,1)
            if predicted == cl:
                a = a - 0.5**i
            else:
                a = a + 0.5**i
        return newImg , a

def any(net, img, attack, clamp=False):
    outputs = net(img)
    _, cl = torch.max(outputs.data,1)
    with torch.no_grad():
        imgLen = img.nelement()
        imShape = img.size()
        x = np.reshape(img.flatten().numpy(),(imgLen,1))
        a = 0.5
        ratio = np.linalg.norm(x) / np.linalg.norm(attack)
        for i in range(1000):
            a = i/1000
            xNew = x +  a * attack * ratio
            if clamp:
                newImg = torch.clamp(torch.tensor(np.reshape(xNew, imShape)),0,1).float()
                d = np.linalg.norm(newImg -img)/np.linalg.norm(x)
            else:
                newImg = torch.tensor(np.reshape(xNew, imShape)).float()
                d=a
            outputs = net(newImg)
            _, predicted = torch.max(outputs.data,1)
            if predicted != cl:
                return newImg, d
        return newImg, 1