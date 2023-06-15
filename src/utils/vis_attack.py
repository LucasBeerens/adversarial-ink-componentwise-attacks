import matplotlib.pyplot as plt
import torch
import numpy as np


def vis_attack(net, img, atts, targeted=True, cl=None, name=None, inverted=False):
    """
    Visualize the results of attacks on a single image using different attack algorithms.

    Args:
    - net: The neural network model.
    - img: The input image.
    - atts: List of attack algorithms to visualize.
    - targeted: Whether the attack is targeted or untargeted.
    - cl: The target class for targeted attacks.
    - name: Name parameter.
    - inverted: Whether to invert the colors of the images (grayscale).
    """

    net.eval()
    output = net(img)
    classCount = output.nelement()
    newImgs = []
    epss = []

    # Compute attacks
    for att in atts:
        if cl is None and not targeted:
            im, eps = att(net, img)
            newImgs.append(im)
            epss.append(eps)
            colCount = 1
        elif targeted and cl is None:
            for c in range(classCount):
                im, eps = att(net, img, c)
                newImgs.append(im)
                epss.append(eps)
            colCount = classCount
        elif targeted and cl is not None:
            im, eps = att(net, img, cl)
            newImgs.append(im)
            epss.append(eps)
            colCount = 1
        else:
            raise Exception('Options are not compatible')

    # Reshape resulting images
    if img.shape[1] == 1:
        cmap = 'gray'
        for i in range(len(newImgs)):
            newImgs[i] = newImgs[i].squeeze()
    elif img.shape[1] == 3:
        cmap = None
        for i in range(len(newImgs)):
            newImgs[i] = np.transpose(newImgs[i].squeeze().numpy(), (1, 2, 0))

    # Create figure and save it
    figure = plt.figure(figsize=(colCount * 2, len(atts) * 2))
    for i, im in enumerate(newImgs):
        ax = figure.add_subplot(len(atts), colCount, i+1)
        ax.title.set_text(round(epss[i], 3))
        outputs = net(im.reshape(img.shape))
        _, predicted = torch.max(outputs.data, 1)
        plt.axis("off")
        if inverted:
            im = torch.ones(28) - im
        plt.imshow(im, cmap=cmap)

    if name is not None:
        plt.savefig('../resultsFinal/{}.png'.format(name), bbox_inches='tight')

    plt.show()
