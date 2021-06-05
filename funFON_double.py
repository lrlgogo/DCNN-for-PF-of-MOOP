import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave

valImageDir = './datSet/FON/dat{0}d/img{0}_{1}.bmp'#datX1d, datX2d
valImageSize = (2**7, 2**7)
valNumImage = 1024
valN = 3
valNumRandom = 100000
valXScale = [-4, 4]
valYScale = [[0, 1], [0, 1]]
valPFScale = [[0, 1], [0, 1]]
# a, b, [min, max]
valParNum = 2
valParScale = [[0.2, 0.8], [0.2, 0.8]]
valRandPar = [1 / (3**0.5), 1 / (3**0.5)]
# FON
valInitiPar = [[1 / (3**0.5), 1 / (3**0.5)],]
valInitiEpi = [[0.01, 0.01],]
valInitiName = ['FON',]

def funFON(x, parSet=valRandPar, parYScale=valYScale):
    a, b = parSet
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    f1 = 1 - np.exp(-sum([(xi - a)**2 for xi in x]))
    f2 = 1 - np.exp(-sum([(xi + b)**2 for xi in x]))
    f1 = np.clip(f1, y1Min, y1Max)
    f2 = np.clip(f2, y2Min, y2Max)
    return f1, f2

def funF(parSet=valRandPar, parXScale=valXScale, parYScale=valYScale,
         parScale=valImageSize, parNumRandom=valNumRandom, n=valN):
    a, b = parSet
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    xScale, y1Scl, y2Scl = xMax - xMin, y1Max - y1Min, y2Max - y2Min
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    xSet = [[xMin + xScale * np.random.random() for i in range(n)]
            for j in range(parNumRandom)]
    for x in xSet:
        f1, f2 = funFON(x, parSet, parYScale)
        f[0].append(int((f1 - y1Min) * (sclMax - sclMin - 1) / y1Scl))
        f[1].append(int((f2 - y2Min) * (sclMax - sclMin - 1) / y2Scl))
    return f

def funPF(parSet=valRandPar, parXScale=valXScale,
          parYScale=valYScale, parScale=valImageSize):
    a, b = parSet
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    xScale, y1Scl, y2Scl = xMax - xMin, y1Max - y1Min, y2Max - y2Min
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    for i in range(sclMax):
        f1 = i * y1Scl / (sclMax - sclMin - 1)
        f1 = np.clip(f1, 0., 1 - 1e-9)
        x = a - (-mt.log(1 - f1) / 3)**0.5
        if (x >= -b) and (x <= a):
            f2 = 1 - mt.exp(-3 * (x + b)**2)
            f[0].append(int(f1 * (sclMax - sclMin - 1) / y1Scl))
            f[1].append(int(f2 * (sclMax - sclMin - 1) / y2Scl))
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
                    parInitiEpi=valInitiEpi, parNumImage=valNumImage,
                    parNum=valParNum):
    x = [
        [funParGenerator_parSet_onePoint(
            parParScale[j][0], parParScale[j][1],
            parInitiEpi[0][j], parInitiPar[0][j]) for j in range(parNum)]
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
