import matplotlib.pyplot as plt

def vis_attack(net, img, target, alg):
    figure = plt.figure(figsize=(8, 8))
    target = 3
    newImg, eps = alg(net,img,target)
    figure.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    figure.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(newImg.squeeze(), cmap="gray")
    plt.show()
    print(eps)