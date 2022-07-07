import matplotlib.pyplot as plt
import torch

def vis_attack(net, img, target, att):
    newImg, eps = att.attack(net,img,target)
    outputs = net(newImg)
    _, predicted = torch.max(outputs.data,1)
    print(predicted)
    print(outputs)
    figure = plt.figure(figsize=(8, 8))
    figure.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    figure.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(newImg.squeeze(), cmap="gray")
    plt.show()
    print(eps)