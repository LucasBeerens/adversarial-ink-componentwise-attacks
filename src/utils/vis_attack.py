import matplotlib.pyplot as plt
import torch
import numpy as np
from data_attacks import attackAny

def vis_attack(net, img, atts, targeted = True, cl = None, name=None):
    net.eval()
    output = net(img)
    classCount = output.nelement()
    newImgs = []
    epss = []


    for att in atts:
        if cl is None and not targeted:
            im, eps = att(net,img)
            newImgs.append(im)
            epss.append(eps)
            colCount = 1
        elif targeted and cl is None:
            for c in range(classCount):
                im, eps = att(net,img,c)
                newImgs.append(im)
                epss.append(eps)
            colCount = classCount
        elif targeted and not cl is None:
            im, eps = att(net,img,cl)
            newImgs.append(im)
            epss.append(eps)
            colCount = 1
        else:
            raise Exception('Options are not compatible')
    
    if img.shape[1] == 1:
        cmap = 'gray'
        for i in range(len(newImgs)):
            newImgs[i] = newImgs[i].squeeze()
    elif img.shape[1] == 3:
        cmap = None
        for i in range(len(newImgs)):
            newImgs[i] = np.transpose(newImgs[i].squeeze().numpy(),(1,2,0))
    
    figure = plt.figure(figsize=(colCount * 2, len(atts)*2))
    for i, im in enumerate(newImgs):
        ax = figure.add_subplot(len(atts), colCount, i+1)
        ax.title.set_text(round(epss[i],3))
        plt.axis("off")
        plt.imshow(im, cmap=cmap)

    if not name is None:
        plt.savefig('../results/{}.png'.format(name),bbox_inches='tight')
    
    plt.show()