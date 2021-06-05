import math as mt
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from scipy.misc import imsave

valImageSize = (2**8, 2**8)
valNumImage = 1024
valXScale = [-3, 3]
valYScale = [9, 81]
valParScale = [-3, 3]

def funSCH(x, parSet=[0, 2]):
    a, b = parSet
    if a > b:
        a, b = b, a
    return (x - a)**2, (x - b)**2

def funPF(parSet=[0, 2], parScale=valImageSize):
    a, b = parSet
    if a > b:
        a, b = b, a
    sclMin, sclMax = valImageSize
    sclMin = 0
    f1pre = 0
    f2pre = (a - b) ** 2
    f1 = []
    f2 = []
    for i in range(sclMin, sclMax):
        y1 = i * valYScale[0] / (sclMax - sclMin - 1)
        if y1 <= (b - a) ** 2:
            y2 = (y1**0.5 + a - b) ** 2
            if y1 == f1pre:
                if y2 < f2pre:
                    f2[-1] = round(y2 / valYScale[1] * (sclMax - sclMin - 1))
            else:
                f1pre = y1
                f2pre = y2
                f1.append(i)
                f2.append(round(y2 / valYScale[1] * (sclMax - sclMin - 1)))
    return f1, f2

def funF(parSet=[0, 2], parScale=valImageSize):
    a, b = parSet
    if a > b:
        a, b = b, a
    sclMin, sclMax = valImageSize
    sclMin = 0
    f1 = []
    f2 = []
    for i in range(sclMin, sclMax):
        y1 = i * valYScale[0] / (sclMax - sclMin - 1)
        y2 = (y1**0.5 + a - b) ** 2
        f1.append(i)
        f2.append(round(y2 / valYScale[1] * (sclMax - sclMin - 1)))
        y2 = (-y1**0.5 + a - b) ** 2
        f1.append(i)
        f2.append(round(y2 / valYScale[1] * (sclMax - sclMin - 1)))
    return f1, f2

"""
x = []
f = [[], []]
'''
for i in range(valXScale[0], valXScale[1]):
    x.append(i)
    f1, f2 = funSCH(i)
    f[0].append(f1)
    f[1].append(f2)
'''
f = funF()
f1, f2 = funPF()
plt.plot(f[0], f[1], 'bo', f1, f2, 'ro')
plt.show()
#"""
#"""
varParSet = []
parMin, parMax = valParScale
i = 0
while i < valNumImage:
    a = np.random.uniform(parMin, parMax)
    b = np.random.uniform(parMin, parMax)
    if a > b:
        a, b = b, a
    if (b - a > 1) and not(abs(a) < 0.5 and abs(b - 2) < 0.3):
        varParSet.append([a, b])
        i += 1

print('parameters sets done.')
varParSet.append([0, 2])

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
        fnameX = './datSet/SCH/datX/' + 'imgX_SCH.bmp'
        fnameY = './datSet/SCH/datY/' + 'imgY_SCH.bmp'
    else:
        fnameX = './datSet/SCH/datX/' + 'imgX_{}.bmp'.format(valImageIndex[i])
        fnameY = './datSet/SCH/datY/' + 'imgY_{}.bmp'.format(valImageIndex[i])
    imsave(fnameX, imgX)
    imsave(fnameY, imgY)

    if i % 50 == 49:
        timEnd = time.time()
        print('Generated Samples: {}, Used {}s.'.format(
            i + 1, timEnd - timStart))
print('GENERATED DONE.')
#"""
