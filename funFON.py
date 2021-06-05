import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave

valImageSize = (2**7, 2**7)
valNumImage = 1024
valXScale = [-4, 4]
valParScale = [0.2, 0.8]
valXRes = 50
valFONPar = [1 / (3**0.5), 1 / (3**0.5)]

def funFON(x, parSet=valFONPar):
    a, b = parSet
    f1 = 1 - np.exp(-sum([(xi - a)**2 for xi in x]))
    f2 = 1 - np.exp(-sum([(xi + b)**2 for xi in x]))
    return f1, f2

def funF(parSet=valFONPar, parScale=valImageSize,
         parXScale=valXScale, parRes=valXRes):
    a, b = parSet
    xMin, xMax = parXScale
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    for i in range(parRes):
        for j in range(parRes):
            for k in range(parRes):
                x1 = xMin + xScale * i / (parRes - 1)
                x2 = xMin + xScale * j / (parRes - 1)
                x3 = xMin + xScale * k / (parRes - 1)
                x = [x1, x2, x3]
                f1, f2 = funFON(x, parSet=parSet)
                f[0].append(int(f1 * (sclMax - sclMin - 1)))
                f[1].append(int(f2 * (sclMax - sclMin - 1)))
    return f
'''
def funF2(parSet=valFONPar, parScale=valImageSize, parXScale=valXScale):
    a, b = parSet
    xMin, xMax = parXScale
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    for i in range(sclMax):
'''

def funPF(parSet=valFONPar, parScale=valImageSize, parXScale=valXScale):
    a, b = parSet
    xMin, xMax = parXScale
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    for i in range(sclMax):
        f1 = i / (sclMax - sclMin - 1)
        f1 = np.clip(f1, 0., 1 - 1e-9)
        x = a - (-mt.log(1 - f1) / 3)**0.5
        if (x >= -b) and (x <= a):
            f2 = 1 - mt.exp(-3 * (x + b)**2)
            f[0].append(int(f1 * (sclMax - sclMin - 1)))
            f[1].append(int(f2 * (sclMax - sclMin - 1)))
    return f
'''
varTemp = [0.11, 1.1]
f = funF(parSet=varTemp)
f12 = funPF(parSet=varTemp)
plt.plot(f[0], f[1], 'bo', ms=1)
plt.plot(f12[0], f12[1], 'r-')
plt.show()
#'''
#"""
varParSet = []
parMin, parMax = valParScale
for i in range(valNumImage):
    varParSet.append([random.uniform(parMin, parMax),
                      random.uniform(parMin, parMax)])
print('Parameters sets done.')
varParSet.append(valFONPar)

valImageIndex = [i for i in range(valNumImage)]
random.shuffle(valImageIndex)
for i in range(valNumImage + 1):
    if i % 50 == 0:
        timStart = time.time()

    f = [[], []]
    f12 = [[], []]
    imgX = 255 * np.ones(valImageSize, dtype='uint8')
    imgY = 255 * np.ones(valImageSize, dtype='uint8')
    f = funF(parSet=varParSet[i])
    f12 = funPF(parSet=varParSet[i])
    for F in zip(f[0], f[1]):
        imgX[valImageSize[1] - 1 - F[1]][F[0]] = 0
    for PF in zip(f12[0], f12[1]):
        imgY[valImageSize[1] - 1 - PF[1]][PF[0]] = 0
    '''
    plt.subplot(121)
    plt.imshow(imgX, cmap='Greys_r')
    plt.subplot(122)
    plt.imshow(imgY, cmap='Greys_r')
    plt.show()
    plt.close()
    '''
    if i == valNumImage:
        fnameX = './datSet/FON/datX/' + 'imgX_FON.bmp'
        fnameY = './datSet/FON/datY/' + 'imgY_FON.bmp'
    else:
        fnameX = './datSet/FON/datX/' + 'imgX_{}.bmp'.format(valImageIndex[i])
        fnameY = './datSet/FON/datY/' + 'imgY_{}.bmp'.format(valImageIndex[i])
    imsave(fnameX, imgX)
    imsave(fnameY, imgY)

    if i % 50 == 49:
        timEnd = time.time()
        print('Generated Samples: {}, Used {}s.'.format(
            i + 1, timEnd - timStart))
print('GENERATED DONE.')
#"""
