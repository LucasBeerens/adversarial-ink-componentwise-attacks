import numpy as np
from utils import conditionNumber
from utils import test_attack
import matplotlib.pyplot as plt


def normwise(net, data, att, name=None):
    """
    Compute normwise condition numbers for dataset and plot it against
    the relative perturbation size.

    It is recommended to use analyse.conditionNumberNormwise instead,
    because test_condition.normwise has to compute the attacks again
    every time you run the function, while analyse.conditionNumberNormwise
    can use saved data.

    Args:
    - net: The neural network model.
    - data: The dataset.
    - att: The attack function to be applied.
    - name: Name parameter for saving the plot.

    Returns:
    - None
    """
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
    """
    Compute componentwise condition numbers for dataset and plot it against
    the relative perturbation size.

    It is recommended to use analyse.conditionNumberComponentwise instead,
    because test_condition.componentwise has to compute the attacks again
    every time you run the function, while analyse.conditionNumberComponentwise
    can use saved data.

    Args:
    - net: The neural network model.
    - data: The dataset.
    - att: The attack function to be applied.
    - name: Name parameter for saving the plot.
    """
    conds = np.array([conditionNumber.componentwise(
        net, img, np.abs(img.flatten())) for img, _ in data])
    eps = test_attack.any(net, data, att).reshape(conds.shape)

    print(np.corrcoef(conds, eps))
    plt.scatter(conds, eps)
    plt.xlabel('Componentwise condition number')
    plt.ylabel('Relative normwise error')

    if name is not None:
        plt.savefig('../results/{}.png'.format(name), bbox_inches='tight')

    plt.show()
