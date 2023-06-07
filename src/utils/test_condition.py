import numpy as np
from utils import conditionNumber
from utils import test_attack
import matplotlib.pyplot as plt


def normwise(net, data, att, name=None):
    conds = np.array([conditionNumber.normwise(net, img) for img, _ in data])
    eps = test_attack.any(net, data, att).reshape(conds.shape)
    print(np.corrcoef(conds, eps))
    plt.scatter(conds, eps)
    plt.xlabel('Normwise condition number')
    plt.ylabel('Relative normwise error')
    if name is not None:
        plt.savefig('../results/{}.png'.format(name), bbox_inches='tight')
    plt.show()


def componentwise(net, data, att, name=None):
    conds = np.array([conditionNumber.componentwise(
        net, img, np.abs(img.flatten())) for img, _ in data])
    eps = test_attack.any(net, data, att).reshape(conds.shape)
    print(np.corrcoef(conds, eps))
    plt.scatter(conds, eps)
    plt.xlabel('Componentwise condition number')
    plt.ylabel('Relative inf norm')
    if name is not None:
        plt.savefig('../results/{}.png'.format(name), bbox_inches='tight')
    plt.show()
