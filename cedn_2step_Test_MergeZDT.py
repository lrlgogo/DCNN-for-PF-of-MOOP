import keras
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing import image
import random
import matplotlib.pyplot as plt

valModelName = ['Model_CEDN_Separation_S1_ZDT12346_128.h5',
                'Model_CEDN_Separation_S2_ZDT12346_128.h5']
valWeightName = ['Weight_Temp_Separation_S1_ZDT12346_128_datXd1.h5',
                 'Weight_Temp_Separation_S2_ZDT12346_128_datYd1.h5']
valImageSize = (2**7, 2**7)

model1 = models.load_model('.\datModels\{}'.format(valModelName[0]))
model1.load_weights('.\datModels\{}'.format(valWeightName[0]))
model1.name += '_model1'
for layer in model1.layers:
    layer.name += '_model1'
model1.summary()
model2 = models.load_model('.\datModels\{}'.format(valModelName[1]))
model2.load_weights('.\datModels\{}'.format(valWeightName[1]))
model2.name += '_model2'
for layer in model2.layers:
    layer.name += '_model2'
model2.summary()
model = models.Sequential()
model.add(model1)
model.add(model2)
model.summary()

"""
from vis.utils import utils

layers_names = [layer.name for layer in model.layers]
last_layer_index = 15#len(layers_names) - 1

from vis.visualization import visualize_activation
import matplotlib.pyplot as plt

img = visualize_activation(model, layer_idx=last_layer_index,
                           max_iter=1000, verbose=False)

img = np.reshape(img, valImageSize)
plt.imshow(img)
plt.show()
"""


def funTestPlot(Xfname, Yfname, parThreshold=0.95, CEDN=model):
    imgX = image.load_img(Xfname, color_mode='grayscale')
    imgTemp = image.load_img(Yfname, color_mode='grayscale')
    imgX = image.img_to_array(imgX)
    imgTemp = image.img_to_array(imgTemp)
    imgTemp = imgTemp.reshape(valImageSize)
    imgX = imgX.astype('float32') / 255
    imgX = imgX.reshape((1,) + np.shape(imgX))
    imgY = CEDN.predict(imgX, verbose=0)
    imgX = np.asarray(imgX, dtype='float32')
    imgX *= 255
    imgX = imgX.reshape(valImageSize)
    imgX = np.clip(imgX, 0, 255).astype('float32')
    plt.subplot(221)
    plt.title('f1-f2')
    
    plt.imshow(imgX, cmap='Greys_r')

    imgY = imgY.reshape(valImageSize)
    
    for j in range(valImageSize[1]):
        varIndex = 0
        varTemp = imgY[varIndex][j]
        for i in range(1, valImageSize[0]):
            if imgY[i][j] < varTemp:
                varTemp = imgY[i][j]
                imgY[varIndex][j] = 1.01
                if imgY[i][j] > parThreshold:
                    imgY[i][j] = 1.01
                else:
                    imgY[i][j] = -0.01
                varIndex = i
            else:
                imgY[i][j] = 1.01

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    metYPred = []
    metYReal = []
    for i in list(range(np.shape(imgY)[0]))[::-1]:
        for j in range(np.shape(imgY)[1]):
            if imgY[np.shape(imgY)[0] - 1 - i][j] < 1:#parThreshold:
                x1.append(j)
                y1.append(i)
                metYPred.append((j / float(valImageSize[1]),
                                 i / float(valImageSize[0])))
            if imgTemp[np.shape(imgY)[0] - 1 - i][j] == 0:
                x2.append(j)
                y2.append(i)
                metYReal.append((j / float(valImageSize[1]),
                                 i / float(valImageSize[0])))
    plt.subplot(223)
    plt.plot(x2, y2, '+', ms=2, label='Theorize')
    plt.plot(x1, y1, 'r.', ms=2, label='Predict')
    plt.legend()
    plt.title('PF Theorize and Predict')
    
    imgY = np.asarray(imgY, dtype='float32')
    imgY *= 255
    imgY = np.clip(imgY, 0, 255).astype('float32')
    imgTemp = np.asarray(imgTemp, dtype='float32')
    imgTemp *= 255
    imgTemp = np.clip(imgTemp, 0, 255).astype('float32')
    plt.subplot(224)
    plt.imshow(imgY, cmap='Greys_r')
    plt.title('PF-Predictional')
    plt.subplot(222)
    plt.imshow(imgTemp, cmap='Greys_r')
    plt.title('PF-Theoretical')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95,
                        hspace=0.25, wspace=0.35)

    metUpisiList = []
    metDeltList = []
    for xx in metYPred:
        metUpisiTemp = []
        if xx != metYPred[-1]:
            x1, y1 = xx
            x2, y2 = metYPred[metYPred.index(xx) + 1]
            metDeltList.append(((x1 - x2)**2 + (y1 - y2)**2)**0.5)
        for yy in metYReal:
            x1, y1 = xx
            x2, y2 = yy
            metUpisiTemp.append(((x1 - x2)**2 + (y1 - y2)**2)**0.5)
        metUpisiList.append(min(metUpisiTemp))
    xx1, yy1 = metYPred[0]
    xx2, yy2 = metYReal[0]
    dl = ((xx1 - xx2)**2 + (yy1 - yy2)**2)**0.5
    xx1, yy1 = metYPred[-1]
    xx2, yy2 = metYReal[-1]
    df = ((xx1 - xx2)**2 + (yy1 - yy2)**2)**0.5
    delt_mean = np.mean(metDeltList)
    Delt = (df + dl + sum([abs(di - delt_mean) for di in metDeltList])
            ) / (df + dl + (len(metDeltList) - 1) * delt_mean)
    print('average of Upisilon:', np.mean(metUpisiList))
    print('variation of Upisilon:', np.var(metUpisiList))
    print('\nvalue of Delta:', Delt)
    
    plt.show()
    plt.close()
    
    return 0

# Xfname = './datSet/SCH/datX/imgX_SCH.bmp'
# Yfname = './datSet/SCH/datY/imgY_SCH.bmp'
""""
for i in range(10):
    Xfname = './datSet/ZDT/ZDT123/datX1d_10d/imgX_{}.bmp'.format(500 + i)
    Yfname = './datSet/ZDT/ZDT123/datZ1d_10d/imgZ_{}.bmp'.format(500 + i)
    funTestPlot(Xfname, Yfname)
"""
for i in ['ZDT1', 'ZDT2', 'ZDT3']:
    print('=======    {}    ======='.format(i))
    Xfname = './datSet/ZDT/ZDT123/datX1d_20d/imgX_{}.bmp'.format(i)
    Yfname = './datSet/ZDT/ZDT123/datZ1d_20d/imgZ_{}.bmp'.format(i)
    funTestPlot(Xfname, Yfname)
