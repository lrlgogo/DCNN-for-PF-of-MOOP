import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave

valImageSize = (2**7, 2**7)
valNumImage = 1# * 5120#2048 * 5

valN = 10#30#10
valXScale = [[0, 1], [0, 1], [0, 1]]
valYScale = [[0, 1], [0, 1]]
valRes = [1*10**2, 1*10**2]

valAScale = [0.4, 2.1]
valBScale = [0, 15]
valCScale = [5, 11]

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

def funZDT4(x, parLis, n=valN):
    a, b, c = parLis
    x1 = x[0]
    f1 = x1
    g = lambda x : 1 + b * (n - 1) + sum([xi**2 - b * mt.cos(c * mt.pi * xi)
                                          for xi in x[1:]])
    gx = g(x)
    f2 = gx * (1 - (x1 / gx)**a)
    return f1, f2

def funZDT6(x, parLis, n=valN):
    a, b, c, d, e, f = parLis
    f1 = 1 - ((mt.sin(b * mt.pi * x[0])) ** a) * mt.exp(-1 * c * x[0])
    g = lambda x : 1 + e * ((sum(x[1:]) / (n - 1)) ** f)
    gx = g(x)
    f2 = gx * (1 - (f1 / gx) ** d)
    return f1, f2

def funPF123(f1, parLis):
    a, b, c = parLis
    return 1 - f1 ** a - f1 * mt.sin(b * mt.pi * f1)

def funPF4(f1, parLis):
    a, _, _ = parLis
    return 1 - f1 ** a

def funPF6(f1, parLis):
    _, _, _, d, _, _ = parLis
    return 1 - f1 ** d

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

#"""
random.seed()
varNumSplit = 0#27#0#9
varXRes = 1000
varSubClassNum = 2108
#varParSet = [[1.5, 12, 5]]
"""
varA = [valAScale[0] + i * (valAScale[1] - valAScale[0]) / (20 - 1)
        for i in range(20)]
#varB = [valBScale[0] + i * (valBScale[1] - valBScale[0]) / (10 - 1)
        #for i in range(10)]
varC = [valCScale[0] + i * (valCScale[1] - valCScale[0]) / (20 - 1)
        for i in range(20)]
varParSet = [[varA[random.randint(0, 19)],
              random.randint(0, 15),
              varC[random.randint(0, 19)]]
             for i in range(valNumImage)]
"""
"""
varParSet = [[random.uniform(valAScale[0], valAScale[1]),
              0,
              random.uniform(valCScale[0], valCScale[1])]
             for i in range(varSubClassNum)] + \
             [[random.uniform(valAScale[0], valAScale[1]),
               random.randint(valBScale[0], valBScale[1]),
               random.uniform(valCScale[0], valCScale[1])]
              for i in range(valNumImage - varSubClassNum)]
#"""
"""ZDT4
varParSet = [[0.3 + (1.3 - 0.3) * np.random.random(),
              8 + (12 - 8) * np.random.random(),
              np.random.randint(2, 5)]
             for i in range(valNumImage)]
#"""
#              a  b   c  d  e   f
varParSet = [[6, 4, 4, 2, 9, 0.25]]
"""
varParSet = [[6, 3 + (5 - 3) * np.random.random(),
              2 + (6 - 2) * np.random.random(),
              0.5 + (2.5 - 0.5) * np.random.random(),
              8 + (10 - 8) * np.random.random(),
              0.2 + (0.3 - 0.2) * np.random.random()]
             for i in range(valNumImage)]
#"""
random.shuffle(varParSet)
valImgIndex = [i for i in range(valNumImage)]
random.shuffle(valImgIndex)
for i in range(valNumImage):
    #"""
    if i % 100 == 0:
        timStart = time.time()
    #"""
    f1 = []
    f2 = []
    f12 = []
    imgX = np.ones(valImageSize, dtype='uint8')
    imgY = np.ones(valImageSize, dtype='uint8')
    """
    xi = 1 / varXRes
    xMinLocal = []
    ftPre = funD1Test(xi, varParSet[i])
    for ii in range(1, varXRes):
        xi += 1 / varXRes
        xi = np.clip(xi, 1e-7, 1.0)
        ftSuc = funD1Test(xi, varParSet[i])
        if (ftPre <= 0) and (ftSuc > 0) and (funD2Test(xi, varParSet[i]) > 0):
            xMinLocal.append(xi)
        ftPre = ftSuc
    #"""
    varEpochs = 0
    varYPreMax = 12#320#2.5
    varYPreMin = 4#0#1.7#-1
    varPFYPreMax = 1
    varPFYPreMin = 0#-1#0
    while varEpochs < 4000:#16000:#2000:
        #"""
        x = [np.random.random() for i in range(valN)]
        #"""
        """
        x = [np.random.random(), np.random.random()] + \
            [0.03 * np.random.random() for i in range(valN - 2)]
        """
        """
        x = [np.random.random() for i in range(valN - varNumSplit)] + \
            [0.05 * np.random.random() for i in range(varNumSplit)]
        #"""
        """
        x = [np.random.random()] + \
            [-5 + 5 * np.random.random()
             for i in range(valN - 1 - varNumSplit)] + \
            [-0.025 + 0.025 * np.random.random() for i in range(varNumSplit)]
        #"""
        """
        x = [np.random.random() for i in range(valN - varNumSplit)] + \
            [np.random.random() for i in range(varNumSplit)]
        """
        """
        x = [np.random.random()] + [-5 + 10 * np.random.random()
                                    for i in range(valN - 1)]
        """
        #y1, y2 = funZDT123(x, varParSet[i])
        #y1, y2 = funZDT4(x, varParSet[i])
        y1, y2 = funZDT6(x, varParSet[i])

        if y2 <= varYPreMax and y2 >= varYPreMin:
            varEpochs += 1
            #if varEpochs % 100 == 0:
                #print(varEpochs)
            #y12 = funPF123(y1, varParSet[i])
            #if y12 >varPFYPreMax:
               # y12 = varPFYPreMax
            #y12 = funPF4(y1, varParSet[i])
            #y12 = funPF6(y1, varParSet[i])
            #"""
            y2 = (y2 - varYPreMin) / (varYPreMax - varYPreMin)
            #y12 = (y12 - varPFYPreMin) / (varPFYPreMax - varPFYPreMin)
            y1 = int(round(y1 * (valImageSize[0] - 1)))
            y2 = int(round(y2 * (valImageSize[1] - 1)))
            #y12 = int(round(y12 * (valImageSize[1] - 1)))
            #"""
            f1.append(y1)
            f2.append(y2)
            #f12.append(y12)
            
            imgX[valImageSize[1] - 1 - y2][y1] = 0
            #imgY[valImageSize[1] - 1 - y12][y1] = 0
    #"""
    for f1x in range(valImageSize[0]):
        f1in = f1x / (valImageSize[0] - 1)
        #y12 = funPF123(f1in, varParSet[i])
        y12 = funPF6(f1in, varParSet[i])
        #y12 = funPF4(f1in, varParSet[i])
        if y12 > varPFYPreMax:
            y12 = varPFYPreMax
        y12 = (y12 - varPFYPreMin) / (varPFYPreMax - varPFYPreMin)
        y12 = int(round(y12 * (valImageSize[1] - 1)))
        f12.append(y12)
        imgY[valImageSize[1] - 1 - y12][f1x] = 0
    #"""
    """
    plt.subplot(121)
    plt.plot(f1, f2, 'o', ms=1, label='(f1, f2)')
    plt.subplot(122)
    plt.plot(f1, f12, 'ro', ms=1, label='(f1, f12)')
    plt.xlim(0, 1)
    #plt.ylim(varYPreMin, varYPreMax)
    plt.show()
    #"""
    """
    #f11 = f1[:]
    f11 = [temp for temp in range(valImageSize[0])]
    for ii in range(len(xMinLocal)):
        xi = xMinLocal[ii]
        xii = int(round(xi * (valImageSize[0] - 1)))
        ftMin = funPF123(xi, varParSet[i])
        ftMin = (ftMin - varPFYPreMin) / (varPFYPreMax - varPFYPreMin)
        ftMin = int(round(ftMin * (valImageSize[1] - 1)))
        kk = 0
        jj = 0
        ww = len(f11)
        while kk < ww:
            kk += 1
            if (f11[jj] > xii) and (f12[jj] >= ftMin):
                imgY[valImageSize[1] - 1 - f12[jj]][f11[jj]] = 1.01
                temp = f11.pop(jj)
                temp = f12.pop(jj)
                jj -= 1
            else:
                imgY[valImageSize[1] - 1 - f12[jj]][f11[jj]] = -0.01
            jj += 1
    #"""
    #"""
    imgX = np.reshape(imgX, valImageSize)
    imgY = np.reshape(imgY, valImageSize)
    imgX *= 255
    imgY *= 255
    imgX = np.clip(imgX, 0, 255).astype('uint8')
    imgY = np.clip(imgY, 0, 255).astype('uint8')
    #"""
    """
    plt.imshow(imgX, cmap='Greys_r')
    plt.figure()
    plt.imshow(imgY, cmap='Greys_r')
    plt.show()
    """
    #"""
    fnameX = './datSet/ZDT/ZDT6/datX/' + 'imgX_{}.bmp'.format('ZDT6')#valImgIndex[i])
    fnameY = './datSet/ZDT/ZDT6/datY/' + 'imgY_{}.bmp'.format('ZDT6')#valImgIndex[i])
    imsave(fnameX, imgX)
    imsave(fnameY, imgY)

    if i % 100 == 99:
        timEnd = time.time()
        print('Generated Samples: {}, Used {}s.'.format(
            i + 1, timEnd - timStart))
    #"""
