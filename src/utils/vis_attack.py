import matplotlib.pyplot as plt
import torch
import numpy as np

def vis_attack(net, img, att, cl = None):
    if cl is None:
        newImg, eps = att(net,img)
    else:
        newImg, eps = att(net,img, cl)

    if img.shape[1] == 1:
        cmap = 'gray'
        im1 = img.squeeze()
        im2 = newImg.squeeze()
    elif img.shape[1] == 3:
        cmap = None
        im1 =np.transpose(img.squeeze().numpy(),(1,2,0))
        im2 = np.transpose(newImg.squeeze().numpy(),(1,2,0))

    outputs = net(newImg)
    _, predicted = torch.max(outputs.data,1)
    print(predicted)
    print(outputs)
    figure = plt.figure(figsize=(8, 8))
    figure.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(im1, cmap=cmap)
    figure.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(im2, cmap=cmap)
    plt.show()
    print(eps)