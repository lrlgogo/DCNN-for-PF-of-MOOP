import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from imageio import imwrite as imsave

valImageDir = './datSet/ZDT/ZDT123/dat{0}1d_20d/img{0}_{1}.bmp'#datX1d, datX2d
valImageSize = (2**7, 2**7)
valNumImage = 1 * 512
valNumRandom = 2000
valN = 20
valXScale = [0, 1]
valYScale = [[0, 1], [0, 10]]
valPFScale = [[0, 1], [-1, 1.1]]
# a, b, c, [min, max]
valParScale = [[0.4, 2.1], [0, 12], [7, 11]]
valRandPar = [0.8, 8, 8]
# ZDT1, ZDT2, ZDT3
valInitiPar = [[0.5, 0, 9], [2, 0, 9], [0.5, 10, 9]]
valInitiName = ['ZDT1', 'ZDT2', 'ZDT3']

def funZDT123(x, parSet=valRandPar, parYScale=valYScale, n=valN):
    a, b, c = parSet
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    x1 = x[0]
    f1 = x1
    g = lambda x : 1 + c * sum(x[1: ]) / (n - 1)
    gx = g(x)
    f2 = gx * (1 - (x1 / gx) ** a - x1 * mt.sin(b * mt.pi * x1) / gx)
    f1 = np.clip(f1, y1Min, y1Max)
    f2 = np.clip(f2, y2Min, y2Max)
    return f1, f2

def funPF123(f1, parSet=valRandPar, parYScale=valYScale):
    a, b, c = parSet
    _, [y2Min, y2Max] = parYScale
    return np.clip(1 - f1 ** a - f1 * mt.sin(b * mt.pi * f1),
                   y2Min, y2Max)

def funD1Test(x, parSet=valRandPar):
    a, b, c = parSet
    return -a * (x ** (a - 1)) \
           -mt.sin(b * mt.pi * x) \
           -x * b * mt.pi * mt.cos(b * mt.pi * x)

def funD2Test(x, parSet=valRandPar):
    a, b, c = parSet
    return -a * (a - 1) * (x ** (a - 2)) \
           -2 * b * mt.pi * mt.cos(b * mt.pi * x) \
           + ((b * mt.pi) ** 2) * x * mt.sin(b * mt.pi * x)

def funF(parSet=valRandPar, parXScale=valXScale, parYScale=valYScale,
         parScale=valImageSize, parNumRandom=valNumRandom, n=valN):
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    y1Scl, y2Scl = y1Max - y1Min, y2Max - y2Min
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    xSet = [[xMin + xScale * np.random.random() for i in range(n)]
            for j in range(parNumRandom)]
    for x in xSet:
        f1, f2 = funZDT123(x, parSet, parYScale, n)
        f[0].append(int((f1 - y1Min) * (sclMax - sclMin - 1) / y1Scl))
        f[1].append(int((f2 - y2Min) * (sclMax - sclMin - 1) / y2Scl))

    return f

def funPF(parSet=valRandPar, parYScale=valPFScale, parScale=valImageSize):
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    y1Scl, y2Scl = y1Max - y1Min, y2Max - y2Min
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    for i in range(sclMax):
        f1 = y1Min + i * y1Scl / (sclMax - sclMin - 1)
        f2 = funPF123(f1, parSet, parYScale)
        f[0].append(int((f1 - y1Min) * (sclMax - sclMin - 1) / y1Scl))
        f[1].append(int((f2 - y2Min) * (sclMax - sclMin - 1) / y2Scl))

    return f

def funPFFilter(parImgX):
    parY = np.copy(parImgX)
    parPF = []
    for i in range(parY.shape[1]):
        for j in range(parY.shape[0]):
            if parY[parY.shape[0] - j - 1][i] == 0:
                parPF.append([i, j])
    
    parPop = []
    for i in range(len(parPF) - 1):
        for j in range(i + 1, len(parPF)):
            if ((parPF[i][0] < parPF[j][0]) and (parPF[i][1] < parPF[j][1])) or\
               ((parPF[i][0] <= parPF[j][0]) and (parPF[i][1] < parPF[j][1])) or\
               ((parPF[i][0] < parPF[j][0]) and (parPF[i][1] <= parPF[j][1])):
                parPop.append(j)
            if ((parPF[i][0] > parPF[j][0]) and (parPF[i][1] > parPF[j][1])) or\
               ((parPF[i][0] >= parPF[j][0]) and (parPF[i][1] > parPF[j][1])) or\
               ((parPF[i][0] > parPF[j][0]) and (parPF[i][1] >= parPF[j][1])):
                parPop.append(i)
    parPop = set(parPop)
    parPop = list(parPop)
    parPop.sort()
    for i in range(len(parPop)):
        parPop[i] -= i
    for popIndex in parPop:
        parPF.pop(popIndex)
    
    return parPF

def funPFTrue(parSet=valRandPar, parXScale=valXScale, parYScale=valYScale,
              parScale=valImageSize, n=valN):
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    y1Scl, y2Scl = y1Max - y1Min, y2Max - y2Min
    xScale = xMax - xMin
    _, sclMax = parScale
    sclMin = 0
    
"""=========================================================================="""
#varSubClassNum = 2108
varParSet = []
varParSet = [
    [random.uniform(valParScale[j][0], valParScale[j][1])
     for j in range(3)]
    for i in range(valNumImage)]
#for i in range(varSubClassNum):
    #varParSet[i][1] = 0
varParSet += valInitiPar
print('Parameters sets done.')

valImageIndex = [i for i in range(valNumImage)]
random.shuffle(valImageIndex)

for i in range(valNumImage + 3):

    if i % 100 == 0:
        timStart = time.time()
        
    f = [[], []]
    f12 = [[], []]
    imgX = 255 * np.ones(valImageSize, dtype='uint8')
    imgY = 255 * np.ones(valImageSize, dtype='uint8')
    imgZ = 255 * np.ones(valImageSize, dtype='uint8')

    f = funF(parSet=varParSet[i])
    f12 = funPF(parSet=varParSet[i])

    for F in zip(f[0], f[1]):
        imgX[valImageSize[1] - 1 - F[1]][F[0]] = 0
    for xPF in zip(f12[0], f12[1]):
        imgZ[valImageSize[1] - 1 - xPF[1]][xPF[0]] = 0
        for index in range(xPF[1], valImageSize[1]):
            imgY[valImageSize[1] - 1 - index][xPF[0]] = 0
    varTemp = funPFFilter(imgZ)
    imgZ = 255 * np.ones(valImageSize, dtype='uint8')
    for xPF in varTemp:
        imgZ[valImageSize[1] - 1 - xPF[1]][xPF[0]] = 0

    if i >= valNumImage:
        fnameX = valImageDir.format('X', valInitiName[i - valNumImage])
        fnameY = valImageDir.format('Y', valInitiName[i - valNumImage])
        fnameZ = valImageDir.format('Z', valInitiName[i - valNumImage])
    else:
        fnameX = valImageDir.format('X', valImageIndex[i])
        fnameY = valImageDir.format('Y', valImageIndex[i])
        fnameZ = valImageDir.format('Z', valImageIndex[i])

    imsave(fnameX, imgX)
    imsave(fnameY, imgY)
    imsave(fnameZ, imgZ)

    if i % 100 == 99:
        timEnd = time.time()
        print('Generated Samples: {}, Used {}s.'.format(
            i + 1, timEnd -timStart))
print('\nGENERATED DONE.')
