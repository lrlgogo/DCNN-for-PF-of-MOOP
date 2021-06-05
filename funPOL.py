import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave

valImageSize = (2**8, 2**8)
valNumImage = 1024
valXScale = [-mt.pi, mt.pi]
valYScale = [30, 60]
valParScale = [0.5, 4]
valXRes = 120
valNumRandom = 5000
valPOLPar = [3, 1]

def funPOL(x, parSet=valPOLPar, parYScale=valYScale):
    a, b = parSet
    y1Scl, y2Scl = parYScale
    x1, x2 = x
    A1 = 0.5 * b * mt.sin(1) - 2 * mt.cos(1) + mt.sin(2) - 0.5 * a * mt.cos(2)
    A2 = 0.5 * a * mt.sin(1) - mt.cos(1) + 2 * mt.sin(2) - 0.5 * b * mt.cos(2)
    B1 = 0.5 * b * mt.sin(x1) - 2 * mt.cos(x1) + mt.sin(x2) - 0.5 * a * mt.cos(x2)
    B2 = 0.5 * a * mt.sin(x1) - mt.cos(x1) + 2 * mt.sin(x2) - 0.5 * b * mt.cos(x2)

    f1 = 1 + (A1 - B1)**2 + (A2 - B2)**2
    f2 = (x1 + a)**2 + (x2 + b)**2

    f1 = np.clip(f1, 0, y1Scl)
    f2 = np.clip(f2, 0, y2Scl)
    
    return f1, f2

def funF(parSet=valPOLPar, parScale=valImageSize,
         parXScale=valXScale, parYScale=valYScale, parRes=valXRes):
    a, b = parSet
    xMin, xMax = parXScale
    y1Scl, y2Scl = parYScale
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    for i in range(parRes):
        for j in range(parRes):
            x1 = xMin + xScale * i / (parRes - 1)
            x2 = xMin + xScale * j / (parRes - 1)
            x = [x1, x2]
            f1, f2 = funPOL(x, parSet=parSet)
            f[0].append(int(f1 * (sclMax - sclMin - 1) / y1Scl))
            f[1].append(int(f2 * (sclMax - sclMin - 1) / y2Scl))
    return f

def funFR(parSet=valPOLPar, parScale=valImageSize,
         parXScale=valXScale, parYScale=valYScale, parNum=valNumRandom):
    a, b = parSet
    xMin, xMax = parXScale
    y1Scl, y2Scl = parYScale
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    parXSet = [[xMin + xScale * np.random.random(),
                xMin + xScale * np.random.random()]
               for i in range(parNum)]
    f = [[], []]
    for i in range(parNum):
        x = parXSet[i]
        f1, f2 = funPOL(x, parSet=parSet)
        f[0].append(int(f1 * (sclMax - sclMin - 1) / y1Scl))
        f[1].append(int(f2 * (sclMax - sclMin - 1) / y2Scl))
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
varTemp = valPOLPar#[0.11, 1.1]
f = funF(parSet=varTemp)
#f12 = funPF(parSet=varTemp)
plt.plot(f[0], f[1], 'bo', ms=1)
#plt.plot(f12[0], f12[1], 'r-')
plt.show()
#'''
#"""
varParSet = []
parMin, parMax = valParScale
for i in range(valNumImage):
    varParSet.append([random.uniform(parMin, parMax),
                      random.uniform(parMin, parMax)])
print('Parameters sets done.')
varParSet.append(valPOLPar)

valImageIndex = [i for i in range(valNumImage)]
random.shuffle(valImageIndex)
for i in range(valNumImage + 1):
    if i % 50 == 0:
        timStart = time.time()

    f = [[], []]
    f12 = []
    imgX = 255 * np.ones(valImageSize, dtype='uint8')
    imgY = 255 * np.ones(valImageSize, dtype='uint8')
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
        fnameX = './datSet/POL/datX/' + 'imgX_POL.bmp'
        fnameY = './datSet/POL/datY/' + 'imgY_POL.bmp'
    else:
        fnameX = './datSet/POL/datX/' + 'imgX_{}.bmp'.format(valImageIndex[i])
        fnameY = './datSet/POL/datY/' + 'imgY_{}.bmp'.format(valImageIndex[i])
    imsave(fnameX, imgX)
    imsave(fnameY, imgY)
    #'''
    if i % 50 == 49:
        timEnd = time.time()
        print('Generated Samples: {}, Used {}s.'.format(
            i + 1, timEnd - timStart))
print('GENERATED DONE.')
#"""
