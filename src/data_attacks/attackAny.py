import numpy as np
import torch

class Any:
    def __init__(self, alg) -> None:
        self.alg = alg

    def __call__(self,net,img):
        outputs = net(img)
        _, cl = torch.max(outputs.data,1)
        
        outputSize = net(img).size(-1)
        resultEps = np.ones(outputSize-1)
        resultImg = []
        for c in range(outputSize-1):
            target = c + (c>=cl)
            newImg, eps = self.alg(net,img,target)
            resultEps[c] = eps
            resultImg.append(newImg)
        bestEps = np.min(resultEps)
        index = np.argmin(resultEps)
        bestImg = resultImg[index]
        return bestImg, bestEps