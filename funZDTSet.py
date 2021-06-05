import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from keras.preprocessing import image

valImageSize = (2**7, 2**7)
valNumImage = 20#2048 * 5

valN = 30
valXScale = [[0, 1], [0, 1], [0, 1]]
valYScale = [[0, 1], [0, 1]]
valRes = [1*10**2, 1*10**2]

valAScale = [0.2, 2.1]
valBScale = [0, 15]
valCScale = [0, 15]

def g1(x, n=valN):
    return 1 + 9 * sum(x[1: ]) / (n - 1)

def funZDT1(x, n=valN):
    x1 = x[0]
    f1 = x1
    gx = g1(x, n)
    f2 = gx * (1 - (x1 / gx) ** 0.5)
    return f1, f2

def funPF1(f1):
    return 1 - f1 ** 0.5

def funZDT123(x, parLis, n=valN):
    a, b, c = parLis
    x1 = x[0]
    f1 = x1
    g = lambda x : 1 + c * sum(x[1: ]) / (n - 1)
    gx = g(x)
    f2 = gx * (1 - (x1 / gx) ** a - x1 * mt.sin(b * mt.pi * x1) / gx)
    return f1, f2

def funPF123(f1, parLis):
    a, b, c = parLis
    return 1 - f1 ** a - f1 * mt.sin(b * mt.pi * f1)

def funD1Test(x, parLis):
    a, b, c = parLis
    return -a * (x ** (a - 1)) \
           -mt.sin(b * mt.pi * x) \
           -x * b * mt.pi * mt.cos(b * mt.pi * x)

def funD2Test(x, parLis):
    a, b, c = parLis
    return -a * (a - 1) * (x ** (a - 2)) \
           -2 * b * mt.pi * mt.cos(b * mt.pi * x) \
           + ((b * mt.pi) ** 2) * x * mt.sin(b * mt.pi * x)

"""
f1 = []
f2 = []
f0 = []
varParSet = [0.1, 10, 5]
for p1 in range(valRes[0]):
    for p2 in range(valRes[1]):
        x = [0.01 * p1, 0.01 * p2]
        y = funZDT123(x, varParSet)
        f1.append(y[0])
        f2.append(y[1])
        f0.append(funPF123(y[0], varParSet))

plt.plot(f1, f2, 'o', f1, f0, 'r', label='ZDT1')
plt.show()
"""
#"""
random.seed()
varXRes = 1000
#varParSet = [0.1, 20, 15]
"""
varParSet = [[random.uniform(valAScale[0], valAScale[1]),
              random.randint(valBScale[0], valBScale[1]),
              random.uniform(valCScale[0], valCScale[1])]
             for i in range(valNumImage)]
"""
varA = [0.2 + i * 0.05 for i in range(40)]
varB = [i for i in range(16)]
varC = [i for i in range(16)]
varParSet = [[varA[i], varB[j], varC[k]]
             for i in range(40)
             for j in range(16)
             for k in range(16)]
random.shuffle(varParSet)
valImgIndex = [i for i in range(valNumImage)]
#random.shuffle(valImgIndex)
for i in range(valNumImage):
    if i % 1000 == 0:
        timStart = time.time()
    f1 = []
    f2 = []
    f12 = []
    imgX = np.ones(valImageSize, dtype='uint8')
    imgY = np.ones(valImageSize, dtype='uint8')
    xi = 1 / varXRes
    xMinLocal = []
    ftPre = funD1Test(xi, varParSet[i])
    for ii in range(1, varXRes):
        xi += 1 / varXRes
        ftSuc = funD1Test(xi, varParSet[i])
        if (ftPre <= 0) and (ftSuc > 0) and (funD2Test(xi, varParSet[i]) > 0):
            xMinLocal.append(xi)
        ftPre = ftSuc
    varEpochs = 0
    varYPreMax = 1 + 0.5 * varParSet[i][2]
    while varEpochs < 2000:
        x = [np.random.random() for i in range(valN)]
        y1, y2 = funZDT123(x, varParSet[i])

        if y2 <= varYPreMax:
            varEpochs += 1
            y12 = funPF123(y1, varParSet[i])
            y2 = (y2 + 1) / (varYPreMax + 1)
            y12 = (y12 + 1) / (varYPreMax + 1)
            y1 = int(round(y1 * (valImageSize[0] - 1)))
            y2 = int(round(y2 * (valImageSize[1] - 1)))
            y12 = int(round(y12 * (valImageSize[1] - 1)))
            f1.append(y1)
            f2.append(y2)
            f12.append(y12)
            
            imgX[valImageSize[1] - 1 - y2][y1] = 0
    f11 = f1[:]
    for ii in range(len(xMinLocal)):
        xi = xMinLocal[ii]
        xii = int(round(xi * (valImageSize[0] - 1)))
        ftMin = funPF123(xi, varParSet[i])
        ftMin = (ftMin + 1) / (varYPreMax + 1)
        ftMin = int(round(ftMin * (valImageSize[1] - 1)))
        kk = 0
        jj = 0
        ww = len(f11)
        while kk < ww:
            kk += 1
            if (f11[jj] > xii) and (f12[jj] >= ftMin):
                imgY[valImageSize[1] - 1 - f12[jj]][f11[jj]] = 1
                f11.pop(jj)
                f12.pop(jj)
                jj -= 1
            else:
                imgY[valImageSize[1] - 1 - f12[jj]][f11[jj]] = 0
            jj += 1
    imgX *= 255
    imgY *= 255
    imgX = np.clip(imgX, 0, 255).astype('uint8')
    imgY = np.clip(imgY, 0, 255).astype('uint8')

    
    fnameX = './datSet/ZDT/datX/' + 'imgX_{}.bmp'.format(valImgIndex[i])
    imsave(fnameX, imgX)
    fnameY = './datSet/ZDT/datY/' + 'imgY_{}.bmp'.format(valImgIndex[i])
    imsave(fnameY, imgY)

    if i % 1000 == 999:
        timEnd = time.time()
        print('Generated Samples: {}, Used {}s.'.format(
            i + 1, timEnd - timStart))
"""
    plt.plot(f1, f2, 'o', f11, f12, 'ro', label='ZDT')
    if i < valNumImage - 1:
        plt.figure()
plt.show()
"""
