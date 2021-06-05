import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from imageio import imwrite as imsave

valImageDir = './datSet/SCH/dat{0}d1/img{0}_{1}.bmp'#datX1d, datX2d
valImageSize = (2**7, 2**7)
valNumImage = 1024
valNumRandom = 4000
valXScale = [-3, 3]
valYScale = [[0, 20], [0, 20]]
valPFScale = [[0, 20], [0, 20]]
# a, b, [min, max]
valParScale = [[-3, 3], [-3, 3]]
valRandPar = [0, 2.2]
# SCH
valInitiPar = [[0, 2]]
valInitiName = ['SCH']

def funSCH(x, parSet=valRandPar, parYScale=valYScale):
    a, b = parSet
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    if a > b:
        a, b = b, a
    f1, f2 = (x - a)**2, (x - b)**2
    f1 = np.clip(f1, y1Min, y1Max)
    f2 = np.clip(f2, y2Min, y2Max)
    return f1, f2

def funF(parSet=valRandPar, parXScale=valXScale, parYScale=valYScale,
         parScale=valImageSize, parNumRandom=valNumRandom):
    xMin, xMax = parXScale
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    xScale, y1Scl, y2Scl = xMax - xMin, y1Max - y1Min, y2Max - y2Min
    _, sclMax = parScale
    sclMin = 0

    f = [[], []]
    xSet = [xMin + xScale * np.random.random() for j in range(parNumRandom)]
    for x in xSet:
        f1, f2 = funSCH(x, parSet, parYScale)
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
def funPF(parSet=valRandPar, parYScale=valPFScale, parScale=valImageSize):
    a, b = parSet
    if a > b:
        a, b = b, a
    _, sclMax = valImageSize
    sclMin = 0
    [y1Min, y1Max], [y2Min, y2Max] = parYScale
    f1pre = 0
    f2pre = (a - b) ** 2

    f = [[], []]
    for i in range(sclMin, sclMax):
        y1 = i * (y1Max - y1Min) / (sclMax - sclMin - 1)
        if y1 <= (b - a) ** 2:
            y2 = (y1**0.5 + a - b) ** 2
            if y1 == f1pre:
                if y2 < f2pre:
                    f[1][-1] = round(y2 / (y2Max - y2Min) * (sclMax - sclMin - 1))
            else:
                f1pre = y1
                f2pre = y2
                f[0].append(i)
                f[1].append(round(y2 / (y2Max - y2Min) * (sclMax - sclMin - 1)))
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
 
"""=========================================================================="""
varParSet = []
i = 0
while i < valNumImage:
    a = np.random.uniform(valParScale[0][0], valParScale[0][1])
    b = np.random.uniform(valParScale[1][0], valParScale[1][1])
    if a > b:
        a, b = b, a
    if (b - a > 1) and not(abs(a) < 0.5 and abs(b - 2) < 0.3):
        varParSet.append([a, b])
        i += 1
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
