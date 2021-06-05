import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave

valImageSize = (2**7, 2**7)
valNumImage = 1024
valXScale = [-5, 5]
valYScale = [[-25, 0], [-20, 10]]
#                                   a, b, c, d , [min, max]
valParScale = [[8., 12.], [0.1, 0.3], [0.5, 1.2], [3., 7.]]
valXRes = 50
valNumRandom = 500000
valKURPar = [10., 0.2, 0.8, 5.]# a, b, c, d

def funKUR(x, parSet=valKURPar, parYScale=valYScale):
    a, b, c, d = parSet
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    f1 = sum([-a * mt.exp(-b * (x[i]**2 + x[i + 1]**2)**0.5)
              for i in range(2)])
    f2 = sum([abs(x[i])**c + d * mt.sin(x[i]**3)
              for i in range(3)])
    f1 = np.clip(f1, y1Min, y1Max)
    f2 = np.clip(f2, y2Min, y2Max)
    
    return f1, f2

def funF(parSet=valKURPar, parScale=valImageSize,
         parXScale=valXScale, parYScale=valYScale, parRes=valXRes):
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    y1Scl, y2Scl = y1Max - y1Min, y2Max - y2Min
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    xSet = []
    for i in range(parRes):
        for j in range(parRes):
            for k in range(parRes):
                x1 = xMin + xScale * i / (parRes - 1)
                x2 = xMin + xScale * j / (parRes - 1)
                x3 = xMin + xScale * k / (parRes - 1)
                xSet.append([x1, x2, x3])
    for x in xSet:
        f1, f2 = funKUR(x, parSet=parSet)
        f[0].append(int((f1 - y1Min) * (sclMax - sclMin - 1) / y1Scl))
        f[1].append(int((f2 - y2Min) * (sclMax - sclMin - 1) / y2Scl))
    return f

def funFR(parSet=valKURPar, parScale=valImageSize,
         parXScale=valXScale, parYScale=valYScale, parNum=valNumRandom):
    a, b, c, d = parSet
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    y1Scl, y2Scl = y1Max - y1Min, y2Max - y2Min
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    parXSet = [[xMin + xScale * np.random.random()
                for j in range(3)]
               for i in range(parNum)]
    f = [[], []]
    for i in range(parNum):
        x = parXSet[i]
        f1, f2 = funKUR(x, parSet=parSet)
        f[0].append(int((f1 - y1Min) * (sclMax - sclMin - 1) / y1Scl))
        f[1].append(int((f2 - y2Min) * (sclMax - sclMin - 1) / y2Scl))
    return f

def funPF(parImgX):
    parY = np.copy(parImgX)
    parPF = []
    for i in range(parY.shape[1]):
        for j in range(parY.shape[0]):
            if parY[parY.shape[0] - j - 1][i] == 0:
                parPF.append([i, j])
    
    parPop = []
    for i in range(len(parPF) - 1):
        for j in range(i + 1, len(parPF)):
            if (parPF[i][0] < parPF[j][0]) and (parPF[i][1] < parPF[j][1]):
                parPop.append(j)
            if (parPF[i][0] > parPF[j][0]) and (parPF[i][1] > parPF[j][1]):
                parPop.append(i)
    parPop = set(parPop)
    parPop = list(parPop)
    parPop.sort()
    for i in range(len(parPop)):
        parPop[i] -= i
    for popIndex in parPop:
        parPF.pop(popIndex)
    
    return parPF
'''
varParSet = [
    [valParScale[j][0] +
     (valParScale[j][1] - valParScale[j][0]) * np.random.random()
     for j in range(4)
     ] for i in range(10)]
for i in range(3):
    print(varParSet[i])
    #f = funF(parSet=varParSet[i])
    f = funFR(parSet=varParSet[i])
    plt.plot(f[0], f[1], 'bo', ms=1)
    plt.show()
plt.close()
'''
#"""
varParSet = []
varParSet = [
    [random.uniform(valParScale[j][0], valParScale[j][1])
     for j in range(4)]
    for i in range(valNumImage)]
print('Parameters sets done.')
varParSet.append(valKURPar)

valImageIndex = [i for i in range(valNumImage)]
random.shuffle(valImageIndex)
for i in range(valNumImage + 1):
    if i % 10 == 0:
        timStart = time.time()

    f = [[], []]
    f12 = []
    imgX = 255 * np.ones(valImageSize, dtype='uint8')
    imgY = 255 * np.ones(valImageSize, dtype='uint8')
    #f = funFR(parSet=varParSet[i])
    f = funF(parSet=varParSet[i])
    for F in zip(f[0], f[1]):
        imgX[valImageSize[1] - 1 - F[1]][F[0]] = 0
    f12 = funPF(imgX)
    for xPF in f12:
        imgY[valImageSize[1] - 1 - xPF[1]][xPF[0]] = 0
    '''
    plt.subplot(121)
    plt.imshow(imgX, cmap='Greys_r')
    plt.subplot(122)
    plt.imshow(imgY, cmap='Greys_r')
    plt.show()
    plt.close()
    #'''
    #'''
    if i == valNumImage:
        fnameX = './datSet/KUR/datX/' + 'imgX_KUR.bmp'
        fnameY = './datSet/KUR/datY/' + 'imgY_KUR.bmp'
    else:
        fnameX = './datSet/KUR/datX/' + 'imgX_{}.bmp'.format(valImageIndex[i])
        fnameY = './datSet/KUR/datY/' + 'imgY_{}.bmp'.format(valImageIndex[i])
    imsave(fnameX, imgX)
    imsave(fnameY, imgY)
    #'''
    if i % 10 == 9:
        timEnd = time.time()
        print('Generated Samples: {}, Used {}s.'.format(
            i + 1, timEnd - timStart))
print('GENERATED DONE.')
#"""
