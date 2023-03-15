import torch
import cvxpy as cp
import numpy as np
import utils.scaleAttack as scale
from utils import jacobian

class Al:
    def __init__(self, num=False) -> None:
        self.num = num
        pass

    def __call__(self,net,img,cl):
        return img, 0