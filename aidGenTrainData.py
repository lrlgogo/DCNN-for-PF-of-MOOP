import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
import random

"""=== Global Parameters ================================================="""
valImageSize = (2**7, 2**7)
valXYScale = [[0, 100], [0, 100]]
valOvalParaLim = [[50, 80], [50, 80],
                  [20, 50], [20, 50],
                  [mt.pi / 8, 3 * mt.pi / 8]]
valNumImg = 10**4
"""=== Functions ========================================================="""
def funAxisRot(parPoint, parAlpha):
    parPoint = np.asarray(parPoint, dtype='float32')
    invA, invB = [mt.cos(parAlpha), mt.sin(parAlpha)]
    invRotMap = np.asarray([[invA, -invB],
                           [invB, invA]], dtype='float32')
    return list(np.dot(parPoint, invRotMap))

def funOval(parList):
    x0, y0, a, b, alpha = parList
    [x1, y1] = funAxisRot([x0, y0], alpha)
    return lambda point : (point[0] - x1)**2 / a**2 + \
                          (point[1] - y1)**2 / b**2 -1

def funSetPix(parPoint, parFun, parPrec=1e-2):
    if parFun(parPoint) < parPrec:
        return 0
    else:
        return 1

def funSetPixBound(parPoint, parFun, parPrec=2.8e-2):
    if abs(parFun(parPoint)) < parPrec:
        return 0
    else:
        return 1

def funPixToXY(parPix, parXYScale=valXYScale, parImageSize=valImageSize):
    [xMin, xMax], [yMin, yMax] = parXYScale
    xPix, yPix = parPix
    xScale, yScale = parImageSize
    return [xMin + (xMax - xMin) * xPix / (xScale - 1),
            yMin + (yMax - yMin) * yPix / (yScale - 1)]

def funRandTrans(parX, parLim):
    xMin, xMax = parLim
    return xMin + (xMax - xMin) * parX

"""===   MAIN   =========================================================="""
random.seed()

varOvalPara = [[random.random() for i in range(5)] for j in range(valNumImg)]
varOvalPara = [[funRandTrans(varOvalPara[i][j], valOvalParaLim[j])
                for j in range(5)]
               for i in range(valNumImg)]

for varIndexOval in range(valNumImg):
    varImg = []
    varImgBound = []
    for xi in range(valImageSize[0]):
        for yi in range(valImageSize[1]):
            varTemp = funAxisRot(funPixToXY([xi, yi]),
                                 varOvalPara[varIndexOval][-1])
            f = funOval(varOvalPara[varIndexOval])
            varImg.append(funSetPix(varTemp, f))
            varImgBound.append(funSetPixBound(varTemp, f))
            
    varImg = np.asarray(varImg, dtype='uint8')
    varImg *= 255
    varImg = varImg.reshape(valImageSize)
    varImg = np.clip(varImg, 0, 255).astype('uint8')
    
    varImgBound = np.asarray(varImgBound, dtype='uint8')
    varImgBound *= 255
    varImgBound = varImgBound.reshape(valImageSize)
    varImgBound = np.clip(varImgBound, 0, 255).astype('uint8')
    
    """
    varImg = list(varImg)
    varImgBound = list(varImgBound)
    varImg.extend(varImgBound)
    varImg = np.clip(varImg, 0, 255).astype('uint8')
    
    plt.imshow(varImg, cmap='Greys_r')
    plt.show()
    """
    fname = './datOvalSet/regionSet/' + 'ovalRegion_{}.bmp'.format(varIndexOval)
    imsave(fname, varImg)
    fname = './datOvalSet/boundSet/' + 'ovalBound_{}.bmp'.format(varIndexOval)
    imsave(fname, varImgBound)
