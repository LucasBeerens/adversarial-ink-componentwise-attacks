import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from utils import latex


def specificToAny(eps):
    count = eps.shape[1] - 1
    n = eps.shape[0]
    result = np.zeros(n * count)
    for i in range(n):
        for c in range(count):
            result[i * count + c] = eps[i, c]
    return result


def comparePerformance(epss, names, name, lw=5, palOrder=None,
                       hueOrder=None, styleOrder=None, legendOrder=None):
    if hueOrder is None:
        hueOrder = legendOrder
    res = pd.DataFrame(np.transpose(np.vstack(epss)), columns=names)
    res['its'] = res.index
    dfm = res.melt('its', var_name='Algorithm', value_name='eps')
    dfm = dfm.sort_values(by=['Algorithm', 'eps'])
    length = epss.shape[1]
    ecdf = np.tile(1 - np.arange(length) / length, (len(names)))
    dfm['Proportion'] = ecdf
    for alg in names:
        dfAlg = dfm[dfm['Algorithm'] == alg]
        small = dfAlg.loc[dfAlg['eps'] < 0.99, 'Proportion'].min()
        dfAlg.loc[dfAlg['eps'] >= 0.99, 'Proportion'] = small
        dfm[dfm['Algorithm'] == alg] = dfAlg
    sns.set(rc={"figure.figsize": (15, 7.5)})
    sns.set(font_scale=2)
    pal = sns.color_palette()
    if palOrder is not None:
        palOrder = [pal[p] for p in palOrder]
    ax = sns.lineplot(data=dfm, x='eps', y='Proportion', palette=palOrder,
                      hue='Algorithm', lw=5, style='Algorithm',
                      style_order=styleOrder, hue_order=hueOrder)

    ax.set(xlabel='Relative normwise perturbation')
    leg_lines = plt.legend().get_lines()
    for line in leg_lines:
        plt.setp(line, linewidth=lw)
    fig = ax.get_figure()
    fig.savefig('../resultsFinal/{}.png'.format(name), bbox_inches='tight')


def compareImages(imgLists, epssLists, labels, indices, targets,
                  name, inverted=True):
    sns.set(font_scale=2.5)
    colCount = len(imgLists)
    rowCount = len(indices)

    figure = plt.figure(figsize=(3*colCount, 3*rowCount))

    for i, (index, target, label) in enumerate(zip(indices, targets, labels)):
        targetClass = target - (target >= label)
        for j, (imgs, epss) in enumerate(zip(imgLists, epssLists)):
            im = imgs[index, targetClass]
            ax = figure.add_subplot(rowCount, colCount, i*colCount + j + 1)
            ax.title.set_text(round(epss[index, targetClass], 3))
            plt.axis("off")
            if inverted:
                im = np.ones(28) - im
            plt.imshow(im, cmap='gray')

    if name is not None:
        plt.savefig('../resultsFinal/{}.png'.format(name), bbox_inches='tight')


def fullComparisonImages(original, imgLists, epssLists, index, label, name):
    sns.set(font_scale=2.5)
    colCount = 10
    rowCount = len(imgLists)

    figure = plt.figure(figsize=(3*colCount, 3*rowCount))

    for i in range(10):
        target = i
        targetClass = target - (target >= label)
        for j, (imgs, epss) in enumerate(zip(imgLists, epssLists)):
            if target == label:
                im = original[index, targetClass]
                im = np.ones(28) - im
                ax = figure.add_subplot(rowCount, colCount, j*colCount + i + 1)
                ax.title.set_text('Original')
                plt.axis("off")
                plt.imshow(im, cmap='gray')
            else:
                im = imgs[index, targetClass]
                ax = figure.add_subplot(rowCount, colCount, j*colCount + i + 1)
                ax.title.set_text(round(epss[index, targetClass], 3))
                plt.axis("off")
                im = np.ones(28) - im
                plt.imshow(im, cmap='gray')
    if name is not None:
        plt.savefig('../resultsFinal/{}.png'.format(name), bbox_inches='tight')


def table(data, epss, outputSize=10):
    results = np.zeros((outputSize, outputSize))
    bestTargets = np.argmin(epss[:, :9], axis=1)
    for i, entry in enumerate(data):
        _, label = entry
        target = bestTargets[i]
        targetClass = target + (target >= label)
        results[label, targetClass] += 1
    bestClassRatio = np.zeros((outputSize, outputSize))
    for i in range(outputSize):
        for j in range(outputSize):
            bestClassRatio[i, j] = np.round(
                results[i, j]/np.sum(results[i, :]), 2)
    latex.matrix(bestClassRatio)


def conditionNumberComponentwise(net, data, epss, imgs, name=None, type=1):
    with torch.no_grad():
        conds = []
        eps = []
        indices = np.argmin(epss[:, :9], axis=1)

        for i, (img, _) in enumerate(data):
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()

            jac = torch.autograd.functional.jacobian(net, img)
            jac = jac.view(classCount, imgLen)

            B = np.abs(jac) @ np.abs(img.flatten())
            output = net(img).numpy()
            conds.append(np.linalg.norm(B, ord=np.inf) /
                         np.linalg.norm(output, ord=np.inf))
            att = imgs[i, indices[i], :, :]
            dif = np.array([np.abs(att - img.numpy()), np.abs(img.numpy())])

            res = []
            for a, b in zip(np.nditer(dif[0, :, :]), np.nditer(dif[1, :, :])):
                if b != 0:
                    res.append(a/b)
            eps.append(max(res))

        if type == 2:
            eps = np.min(epss[:, :9], axis=1)

        print(np.corrcoef(conds, eps))
        sns.set(font_scale=2)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(conds, eps)
        plt.xlabel('Componentwise condition number')
        if type == 1:
            plt.ylabel('Relative componentwise perturbation')
        else:
            plt.ylabel('Relative normwise perturbation')
        if name is not None:
            plt.savefig('../resultsFinal/{}.png'.format(name),
                        bbox_inches='tight')
        plt.show()


def conditionNumberNormwise(net, data, epss, imgs, name=None, type=1):
    with torch.no_grad():
        conds = []
        eps = []

        for i, (img, _) in enumerate(data):
            output = net(img).numpy().transpose()
            classCount = output.size
            imgLen = img.nelement()

            jac = torch.autograd.functional.jacobian(net, img)
            jac = jac.view(classCount, imgLen)

            outputNorm = np.linalg.norm(output)
            jacNorm = np.linalg.norm(jac, ord=2)
            imNorm = np.linalg.norm(img)
            conds.append(jacNorm * imNorm / outputNorm)

        eps = np.min(epss[:, :9], axis=1)

        print(np.corrcoef(conds, eps))
        sns.set(font_scale=2)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(conds, eps)
        plt.xlabel('Normwisewise condition number')
        plt.ylabel('Relative normwise perturbation')
        if name is not None:
            plt.savefig('../resultsFinal/{}.png'.format(name),
                        bbox_inches='tight')
        plt.show()


def specificToBest(eps):
    count = eps.shape[1] - 1
    res = np.min(eps[:, :count], axis=1)
    return res
