import torch
from utils import scaleAttack as scale
import numpy as np

class Att:
    def __init__(self, alg) -> None:
        self.alg = alg
        pass

    def __call__(self, net, img):
        outputs = net(img)
        _, predicted = torch.max(outputs.data,1)
        label = torch.tensor([predicted])

        adv_image = self.alg(img, label)
        delta = adv_image - img
        delta = torch.reshape(delta, (delta.nelement(),1)).numpy()
        eps = np.linalg.norm(delta) / np.linalg.norm(img)
        return scale.any(net, img, delta)