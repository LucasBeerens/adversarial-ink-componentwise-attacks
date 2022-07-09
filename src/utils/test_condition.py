import numpy as np
from utils import conditionNumber
from utils import test_attack
import matplotlib.pyplot as plt

def normwise(net, data, att):
    conds = np.array([conditionNumber.normwise(net, img) for img, _ in data])
    eps = test_attack.any(net,data,att)
    plt.scatter(conds, eps)
    plt.show()

def componentwise(net, data, att):
    conds = np.array([conditionNumber.componentwise(net, img, np.abs(img.flatten())) for img, _ in data])
    eps = test_attack.any(net,data,att)
    plt.scatter(conds, eps)
    plt.show()