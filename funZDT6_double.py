import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave

valImageDir = './datSet/ZDT/ZDT6/dat{0}d/img{0}_{1}.bmp'#datX1d, datX2d
valImageSize = (2**7, 2**7)
valNumImage = 1 * 5120
valNumRandom = 4000
valN = 10
valXScale = [0, 1]
valYScale = [[0, 1], [4, 12]]
valPFScale = [[0, 1], [0, 1]]
# a, b, c, d, e, f, [min, max]
valParScale = [[6, 6], [3, 5], [2, 6], [0.5, 2.5], [8, 10], [0.2, 0.3]]
valRandPar = [6., 4., 4., 1.5, 9., 0.25]
# ZDT6
valInitiPar = [[6., 4., 4., 2., 9., 0.25],]
valInitiEpi = [[0., 0.2, 0.2, 0.1, 0.2, 0.01],]
valInitiName = ['ZDT6',]

def funZDT6(x, parSet=valRandPar, parYScale=valYScale, n=valN):
    a, b, c, d, e, f = parSet
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    f1 = 1 - ((mt.sin(b * mt.pi * x[0])) ** a) * mt.exp(-1 * c * x[0])
    g = lambda x : 1 + e * ((sum(x[1:]) / (n - 1)) ** f)
    gx = g(x)
    f2 = gx * (1 - (f1 / gx) ** d)
    f1 = np.clip(f1, y1Min, y1Max)
    f2 = np.clip(f2, y2Min, y2Max)
    return f1, f2

def funPF6(f1, parSet=valRandPar, parYScale=valYScale):
    _, _, _, d, _, _ = parSet
    _, [y2Min, y2Max] = parYScale
    return np.clip(1 - f1 ** d, y2Min, y2Max)

def funF(parSet=valRandPar, parXScale=valXScale, parYScale=valYScale,
         parScale=valImageSize, parNumRandom=valNumRandom, n=valN):
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    xScale, y1Scl, y2Scl = xMax - xMin, y1Max - y1Min, y2Max - y2Min
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    xSet = [[xMin + xScale * np.random.random() for i in range(n)]
            for j in range(parNumRandom)]
    for x in xSet:
        f1, f2 = funZDT6(x, parSet, parYScale, n)
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
        f2 = funPF6(f1, parSet, parYScale)
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

def funParGenerator_parSet_onePoint(parXMin, parXMax, parEpi, parPoint):
    x = random.uniform(parXMin, parXMax - 2 * parEpi)
    if x > (parPoint - parEpi):
        x += (2 * parEpi)
    return x
def funParGenerator(parParScale=valParScale, parInitiPar=valInitiPar,
                    parInitiEpi=valInitiEpi, parNumImage=valNumImage):
    x = [
        [funParGenerator_parSet_onePoint(
            parParScale[j][0], parParScale[j][1],
            parInitiEpi[0][j], parInitiPar[0][j]) for j in range(6)]
        for i in range(parNumImage)
        ]
    return x
"""=========================================================================="""
varParSet = funParGenerator()

varParSet += valInitiPar
print('Parameters sets done.')

valImageIndex = [i for i in range(valNumImage)]
random.shuffle(valImageIndex)

for i in range(valNumImage + 1):

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
