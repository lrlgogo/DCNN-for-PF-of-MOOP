import math as mt
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave
import time

valImageSize = (2**7, 2**7)
valNumImage = 2048 * 10

valP = [0.7, 1.0, 1.3]
valN = 3
valA = [0.8, 1.0, 1.3]
valB = [0.6, 0.75, 0.9]
valXScale = [[0, 1], [0, 1], [0, 1]]
#valImageScale = (2**7, 2**7)
valImageScale = 5#1*10**2
valRes = [1 * 10 ** 2, 1 * 10 ** 1, 1 * 10 ** 1]

def g(x, n=valN, a=1.3, b=0.8):
    return 1 + a * ((sum(x) - x[0]) / (n - 1))**b

def funTest(x, p=valP[0], n=valN, a=1.3, b=0.8):
    f1 = x[0]
    gx = g(x, n, b)
    f2 = (gx * (1 - (x[0] / gx)**p))**(1 / p)
    return [f1, f2]

def funTestF(x, p=valP[0], n=valN):
    x1 = x.pop(0)
    f1 = x1
    xx = [xi ** 2 - 10 * mt.cos(4*mt.pi*xi) for xi in x]
    gx = 1 + 10 * (n - 1) + sum(xx)
    f2 = gx * (1 - (x1 / gx) ** p)
    return f1, f2

def funPF(x, p=valP[0]):
    return (1 - x**p)**(1/p)

def funPixToVal(p, valScale, pScale):
    valMin, valMax = valScale
    return valMin + (valMax - valMin) * p / (pScale - 1)

def funValToPix(x, xScale, res):
    xMin, xMax = xScale
    return round(x * res / (xMax - xMin))


"""
for p1 in range(valImageScale):
    for p2 in range(valImageScale):
        for p3 in range(valImageScale):
            for p4 in range(valImageScale):
                for p5 in range(valImageScale):
                    x1, x2, x3, x4, x5 = \
                        funPixToVal(p1, valXScale, valImageScale),\
                        funPixToVal(p2, valXScale, valImageScale),\
                        funPixToVal(p3, valXScale, valImageScale),\
                        funPixToVal(p4, valXScale, valImageScale),\
                        funPixToVal(p5, valXScale, valImageScale)
                    x.append([x1, x2, x3, x4, x5])
                    f1, f2 = funTest(x[-1])
                    y1.append(f1)
                    y2.append(f2)
"""
"""
for p1 in range(100*valImageScale):
    for p2 in range(valImageScale):
        for p3 in range(valImageScale):
            x1, x2, x3 = [funPixToVal(p1, valXScale[0], 100*valImageScale),
                          funPixToVal(p2, valXScale[1], valImageScale),
                          funPixToVal(p3, valXScale[2], valImageScale)]
            x.append([x1, x2, x3])
            f1, f2 = funTest(x[-1])
            if f2 <= 1.0:
                y1.append(f1)
                y2.append(f2)

plt.plot(y1, y2, 'b', label='Test')

plt.show()
"""
"""
img = np.zeros(valImageScale, dtype='uint8')

for p1 in range(valRes[0]):
    for p2 in range(valRes[1]):
        x1, x2 = [funPixToVal(p1, valXScale[0], valRes[0]),
                  funPixToVal(p2, valXScale[1], valRes[1])]
        x.append([x1, x2])
        f1, f2 = funTest(x[-1])
        if f2 <= 1.0:
            y1.append(f1)
            y2.append(f2)
plt.plot(y1, y2, 'o', label='Test')
plt.show()
"""


#"""
valImgIndex = [i for i in range(valNumImage)]
random.shuffle(valImgIndex)
for i in range(valNumImage):
    if i % 32 == 0:
        timStart = time.time()
        print('Generate sample batch: {} ...'.format(i // 32 + 1))
    x = []
    y1 = []
    y2 = []
    yt2 = []
    imgX = np.ones(valImageSize, dtype='uint8')
    imgY = np.ones(valImageSize, dtype='uint8')
    varEpochs = 0
    varParIndex = [random.randint(0, 2) for i in range(3)]
    while varEpochs < 2000:
        x = [np.random.random() for i in range(valN)]
        f1, f2 = funTest(x, valP[varParIndex[0]], valN,
                         valA[varParIndex[1]], valB[varParIndex[2]])
        f2t = funPF(f1, valP[varParIndex[0]])
        f2t = int(round(f2t * (valImageSize[1] - 1)))
        if f2 <= 1.0:
            varEpochs += 1
            #print(varEpochs)
            #if varEpochs % 100 == 0:
                #print(varEpochs)
            f1 = int(round(f1 * (valImageSize[0] - 1)))
            f2 = int(round(f2 * (valImageSize[1] - 1)))
            y1.append(f1)
            y2.append(f2)
            yt2.append(f2t)
            imgX[-f2 + valImageSize[1] - 1][f1] = 0
            imgY[-f2t + valImageSize[1] - 1][f1] = 0
    imgX = np.reshape(imgX, valImageSize)
    imgX *= 255
    imgY *= 255
    imgX = np.clip(imgX, 0, 255).astype('uint8')
    imgY = np.clip(imgY, 0, 255).astype('uint8')
    
    fnameX = './datSet/datX1/' + 'imgX_{}.bmp'.format(valImgIndex[i])
    fnameY = './datSet/datY1/' + 'imgY_{}.bmp'.format(valImgIndex[i])
    imsave(fnameX, imgX)
    imsave(fnameY, imgY)
    if i % 32 == 31:
        print(6 * ' ' + 'Sample batch {} saved.'.format(i // 32 + 1))
        timEnd = time.time()
        print(6 * ' ' + 'Using {}s.'.format(timEnd - timStart))
"""
    plt.imshow(imgX, cmap='Greys_r')
    plt.figure()
    plt.imshow(imgY, cmap='Greys_r')
    plt.show()
"""

#"""
"""
for p1 in range(valRes[0]):
    for p2 in range(valRes[1]):
        for p3 in range(valRes[2]):
            x1, x2, x3 = [funPixToVal(p1, valXScale[0], valRes[0]),
                          funPixToVal(p2, valXScale[1], valRes[1]),
                          funPixToVal(p3, valXScale[2], valRes[2])]
            x.append([x1, x2, x3])
            #f1, f2 = funTestF(x[-1])
            f1, f2 = funTest(x[-1])
            f2t = funPF(f1)
            f2t = round(f2t * (valImageSize[1] - 1))
            if f2 <= 1.0:
                #y1.append(f1)
                #y2.append(f2)
                f1 = round(f1 * (valImageSize[0] - 1))
                f2 = round(f2 * (valImageSize[1] - 1))
                y1.append(f1)
                y2.append(f2)
                yt2.append(f2t)
                imgX[-f2 + valImageSize[1] - 1][f1] = 0
                imgY[-f2t + valImageSize[1] - 1][f1] = 0
"""
"""
plt.plot(y1, y2, 'o', label='Test')
plt.figure()

plt.plot(y1, yt2, 'o', label='Test')
plt.show()
"""
