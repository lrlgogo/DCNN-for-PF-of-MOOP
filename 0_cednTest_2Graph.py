import keras
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing import image
import random
import matplotlib.pyplot as plt

"""==========   DATA   ==================================================="""
varArgList = [
    ['ZDT1',
     ['datX1d', 'datZ1d'],
     (2**7, 2**7),
     ['Model_CEDN_Intergrant_ZDT123_128.h5',
      ['Model_CEDN_Separation_S1_ZDT123_128.h5',
       'Model_CEDN_Separation_S2_ZDT123_128.h5']],
     ['Weight_Intergrant_ZDT123_128_datX1d.h5',
      ['Weight_Separation_S1_ZDT123_128_datX1d.h5',
       'Weight_Separation_S2_ZDT123_128_datY1d.h5']],
     [0.95, 0.5]],
    ['ZDT2',
     ['datX1d', 'datZ1d'],
     (2**7, 2**7),
     ['Model_CEDN_Intergrant_ZDT123_128.h5',
      ['Model_CEDN_Separation_S1_ZDT123_128.h5',
       'Model_CEDN_Separation_S2_ZDT123_128.h5']],
     ['Weight_Intergrant_ZDT123_128_datX1d.h5',
      ['Weight_Separation_S1_ZDT123_128_datX1d.h5',
       'Weight_Separation_S2_ZDT123_128_datY1d.h5']],
     [0.95, 0.5]],
    ['ZDT3',
     ['datX1d', 'datZ1d'],
     (2**7, 2**7),
     ['Model_CEDN_Intergrant_ZDT123_128.h5',
      ['Model_CEDN_Separation_S1_ZDT123_128.h5',
       'Model_CEDN_Separation_S2_ZDT123_128.h5']],
     ['Weight_Intergrant_ZDT123_128_datX1d.h5',
      ['Weight_Separation_S1_ZDT123_128_datX1d.h5',
       'Weight_Separation_S2_ZDT123_128_datY1d.h5']],
     [0.95, 0.5]],
    ['ZDT4',
     ['datXd5', 'datZd5'],
     (2**7, 2**7),
     ['Model_CEDN_Intergrant_ZDT4_128.h5',
      ['Model_CEDN_Separation_S1_ZDT4_128.h5',
       'Model_CEDN_Separation_S2_ZDT123_128.h5']],
     ['Weight_Temp_Intergrant_ZDT4_128_datXd5.h5',#'Weight_Intergrant_ZDT4_128_datXd5.h5',
      ['Weight_Separation_S1_ZDT4_128_datXd5.h5',
       'Weight_Separation_S2_ZDT4_128_datYd5.h5']],
     [0.95, 0.95]],
    ['ZDT6',
     ['datXd', 'datZd'],
     (2**7, 2**7),
     ['Model_CEDN_Intergrant_ZDT6_128.h5',
      ['Model_CEDN_Separation_S1_ZDT6_128.h5',
       'Model_CEDN_Separation_S2_ZDT6_128.h5']],
     ['Weight_Intergrant_ZDT6_128_datXd.h5',
      ['Weight_Separation_S1_ZDT6_128_datXd.h5',
       'Weight_Separation_S2_ZDT6_128_datYd.h5']],
     [0.95, 0.95]],
    ['SCH',
     ['datXd', 'datZd'],
     (2**8, 2**8),
     ['Model_CEDN_Intergrant_SCH_256.h5',
      ['Model_CEDN_Separation_S1_SCH_256.h5',
       'Model_CEDN_Separation_S2_SCH_256.h5']],
     ['Weight_Intergrant_SCH_256_datXd.h5',
      ['Weight_Separation_S1_SCH_256_datXd.h5',
       'Weight_Separation_S2_SCH_256_datYd.h5']],
     [0.95, 0.5]],
    ['FON',
     ['datXd', 'datZd'],
     (2**7, 2**7),
     ['Model_CEDN_Intergrant_FON_128.h5',
      ['Model_CEDN_Separation_S1_FON_128.h5',
       'Model_CEDN_Separation_S2_FON_128.h5']],
     ['Weight_Intergrant_FON_128_datXd.h5',
      ['Weight_Separation_S1_FON_128_datXd.h5',
       'Weight_Separation_S2_FON_128_datYd.h5']],
     [0.95, 0.5]],
    ['POL',
     ['datXd1', 'datZd1'],
     (2**8, 2**8),
     ['Model_CEDN_Intergrant_POL_256.h5',
      ['Model_CEDN_Separation_S1_POL_256.h5',
       'Model_CEDN_Separation_S2_POL_256.h5']],
     ['Weight_Intergrant_POL_256_datXd1.h5',
      ['Weight_Separation_S1_POL_256_datXd1.h5',
       'Weight_Separation_S2_POL_256_datYd1.h5']],
     [0.95, 0.5]],
    ['KUR',
     ['datXd', 'datZd'],
     (2**7, 2**7),
     ['Model_CEDN_Intergrant_KUR_128.h5',
      ['Model_CEDN_Separation_S1_KUR_128.h5',
       'Model_CEDN_Separation_S2_KUR_128.h5']],
     ['Weight_Intergrant_KUR_128_datXd.h5',
      ['Weight_Separation_S1_KUR_128_datXd.h5',
       'Weight_Separation_S2_KUR_128_datYd.h5']],
     [0.95, 0.95]],
    ]
"""==========   FUNCTION   ==============================================="""
def funLoadModelIntegrated(parModelFile, parWeightFile):
    model = models.load_model(parModelFile)
    model.load_weights(parWeightFile)
    #model.summary()

    return model

def funLoadModelSeparated(parModelFile, parWeightFile):
    model_step1_file, model_step2_file = parModelFile
    weight_step1_file, weight_step2_file = parWeightFile

    model_step1 = models.load_model(model_step1_file)
    model_step1.load_weights(weight_step1_file)
    model_step1.name = 'model_step1'
    for layer in model_step1.layers:
        layer.name += '_step1'

    model_step2 = models.load_model(model_step2_file)
    model_step2.load_weights(weight_step2_file)
    model_step2.name = 'model_step2'
    for layer in model_step2.layers:
        layer.name += '_step2'

    model = models.Sequential()
    model.add(model_step1)
    model.add(model_step2)
    #model.summary()

    return model

def funTestPlot(parXFName, parYFName,
                parImageSize, parModel, parThreshold=0.95):
    imgX = image.load_img(parXFName, color_mode='grayscale')
    imgTemp = image.load_img(parYFName, color_mode='grayscale')
    imgX = image.img_to_array(imgX)
    imgTemp = image.img_to_array(imgTemp)
    imgTemp = imgTemp.reshape(parImageSize)
    imgX = imgX.astype('float32') / 255
    imgX = imgX.reshape((1,) + np.shape(imgX))
    imgY = parModel.predict(imgX, verbose=0)
    imgX = np.asarray(imgX, dtype='float32')
    imgX *= 255
    imgX = imgX.reshape(parImageSize)
    imgX = np.clip(imgX, 0, 255).astype('float32')
    plt.figure(figsize=(6, 6))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    #plt.title('f1-f2')8
    plt.xlabel('f1', fontsize=18)
    plt.ylabel('f2', fontsize=18)
    #plt.xticks([])
    #plt.yticks([])
    
    plt.imshow(imgX, cmap='Greys_r')
    plt.show()

    imgY = imgY.reshape(parImageSize)
    
    for j in range(parImageSize[1]):
        varIndex = 0
        varTemp = imgY[varIndex][j]
        for i in range(1, parImageSize[0]):
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
                metYPred.append((j / float(parImageSize[1]),
                                 i / float(parImageSize[0])))
            if imgTemp[np.shape(imgY)[0] - 1 - i][j] == 0:
                x2.append(j)
                y2.append(i)
                metYReal.append((j / float(parImageSize[1]),
                                 i / float(parImageSize[0])))
    plt.figure(figsize=(6, 6))
    plt.plot(x2, y2, '+', ms=2, label='Theoretical PF')
    plt.plot(x1, y1, 'r.', ms=2, label='Prediction  PF')
    plt.xlabel('f1', fontsize=18)
    plt.ylabel('f2', fontsize=18)
    #plt.xticks([])
    #plt.yticks([])
    plt.legend(fontsize=18)
    #plt.title('PF Theorize and Predict')
    
    imgY = np.asarray(imgY, dtype='float32')
    imgY *= 255
    imgY = np.clip(imgY, 0, 255).astype('float32')
    imgTemp = np.asarray(imgTemp, dtype='float32')
    imgTemp *= 255
    imgTemp = np.clip(imgTemp, 0, 255).astype('float32')
    """
    plt.subplot(224)
    plt.imshow(imgY, cmap='Greys_r')
    plt.title('PF-Predictional')
    plt.subplot(222)
    plt.imshow(imgTemp, cmap='Greys_r')
    plt.title('PF-Theoretical')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95,
                        hspace=0, wspace=0.1)
    """

    plt.show()
    plt.close()
    
    return 0

def funSourceFile_subTest(invName, invTrainFileName):
    if invName in ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']:
        if invName in ['ZDT1', 'ZDT2', 'ZDT3']:
            x_file = './datSet/ZDT/ZDT123/{1}/imgX_{0}.bmp'.format(
                invName, invTrainFileName[0])
            y_file = './datSet/ZDT/ZDT123/{1}/imgZ_{0}.bmp'.format(
                invName, invTrainFileName[1])
        else:
            x_file = './datSet/ZDT/{0}/{1}/imgX_{0}.bmp'.format(
                invName, invTrainFileName[0])
            y_file = './datSet/ZDT/{0}/{1}/imgZ_{0}.bmp'.format(
                invName, invTrainFileName[1])
    else:
        x_file = './datSet/{0}/{1}/imgX_{0}.bmp'.format(invName,
                                                        invTrainFileName[0])
        y_file = './datSet/{0}/{1}/imgZ_{0}.bmp'.format(invName,
                                                        invTrainFileName[1])

    return x_file, y_file
def funModelWeightFile_subTest(invModelFile, invWeightFile):
    base_dir = 'D:/workspace/DLMOP/CEDN/datModels/'
    model_integrated_path = base_dir + invModelFile[0]
    model_separated_path = [base_dir + invModelFile[1][0],
                            base_dir + invModelFile[1][1]]
    weight_integrated_path = base_dir + invWeightFile[0]
    weight_separated_path = [base_dir + invWeightFile[1][0],
                             base_dir + invWeightFile[1][1]]

    return [model_integrated_path, model_separated_path], \
           [weight_integrated_path, weight_separated_path]
def subTest(args):
    invName, invTrainFileName, invImageSize, \
             invModelFile, invWeightFile, invThreshold = args
    model_integrated_file, model_separated_file = invModelFile
    weight_integrated_file, weight_separated_file = invModelFile
    x_file, y_file = funSourceFile_subTest(invName, invTrainFileName)
    [model_integrated_path, model_separated_path] , \
                           [weight_integrated_path, weight_separated_path] \
                           = funModelWeightFile_subTest(
                               invModelFile, invWeightFile)

    model_integrated = funLoadModelIntegrated(model_integrated_path,
                                              weight_integrated_path)
    print('\n=====   {0}   =====   Integrated Model   =====\n'.format(invName))
    funTestPlot(x_file, y_file,
                invImageSize, model_integrated, invThreshold[0])
    del model_integrated

    model_separated = funLoadModelSeparated(model_separated_path,
                                            weight_separated_path)
    print('\n=====   {0}   =====   Separated  Model   =====\n'.format(invName))
    funTestPlot(x_file, y_file,
                invImageSize, model_separated, invThreshold[1])
    del model_separated

    return 0

"""==========  MAIN  ====================================================="""
for arg in varArgList:
    subTest(arg)
