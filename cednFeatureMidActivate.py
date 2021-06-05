import keras
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing import image
import random
import matplotlib.pyplot as plt
import os

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
def funLoadModel(parModelFile, parWeightFile):
    model = models.load_model(parModelFile)
    model.load_weights(parWeightFile)
    #model.summary()

    return model

def funTestPlot(parXFName, parYFName, parImageSize, parModel,
                parPath, parThreshold=0.95, parS1Flag=False):
    
    imgX = image.load_img(parXFName, color_mode='grayscale')
    imgTemp = image.load_img(parYFName, color_mode='grayscale')
    imgX = image.img_to_array(imgX)
    imgTemp = image.img_to_array(imgTemp)
    imgTemp = imgTemp.reshape(parImageSize)
    imgX = imgX.astype('float32') / 255
    imgX = imgX.reshape((1,) + np.shape(imgX))
    imgY = parModel.predict(imgX, verbose=0)

    layCount = len(parModel.layers) - 1
    images_per_row = 16
    layer_names = []
    for layer in parModel.layers[: layCount]:
        layer_names.append(layer.name)
    layers_outputs = [layer.output for layer in parModel.layers[: layCount]]
    activation_model = models.Model(inputs=parModel.input,
                                    outputs=layers_outputs)
    activations = activation_model.predict(imgX)
    
    imgX = np.asarray(imgX, dtype='float32')
    imgX *= 255
    imgX = imgX.reshape(parImageSize)
    imgX = np.clip(imgX, 0, 255).astype('float32')
    plt.subplot(221)
    plt.title('f1-f2')
    
    plt.imshow(imgX, cmap='Greys_r')

    imgY = imgY.reshape(parImageSize)

    if not(parS1Flag):
        for j in range(parImageSize[1]):
            varIndex = 0
            varTemp = imgY[varIndex][j]
            for i in range(1, parImageSize[0]):
                if imgY[i][j] < varTemp:
                    varTemp = imgY[i][j]
                    imgY[varIndex][j] = 1.01
                    '''
                    if imgY[i][j] > parThreshold:
                        imgY[i][j] = 1.01
                    else:
                        imgY[i][j] = -0.01
                    '''
                    varIndex = i
                else:
                    imgY[i][j] = 1.01

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in list(range(np.shape(imgY)[0]))[::-1]:
        for j in range(np.shape(imgY)[1]):
            if imgY[np.shape(imgY)[0] - 1 - i][j] < parThreshold:
                x1.append(j)
                y1.append(i)
            if imgTemp[np.shape(imgY)[0] - 1 - i][j] == 0:
                x2.append(j)
                y2.append(i)
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
    parFigPath = os.path.join(parPath, '_Input.png')
    plt.savefig(parFigPath)
    #plt.show()
    plt.close()

    for layer_name, layer_activation, layer_index in zip(
        layer_names, activations, range(layCount)):
        feature_num = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        cols_num = feature_num // images_per_row
        display_grid = np.zeros((size*cols_num, size*images_per_row))
        for col in range(cols_num):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :,
                                                 col*images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= (channel_image.std() + 1e-10)
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
                
        scale = 2. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name, fontsize=20)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='Greys_r')
        parFigPath = os.path.join(
            parPath, str(layer_index) + '_' + layer_name + '_midAct.png')
        plt.savefig(parFigPath, bbox_inches='tight')
        plt.close()
    #plt.show()
    #plt.close()

def funSourceFile_subTest(invName, invTrainFileName):
    if invName in ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']:
        if invName in ['ZDT1', 'ZDT2', 'ZDT3']:
            x_file = './datSet/ZDT/ZDT123/{1}/imgX_{0}.bmp'.format(
                invName, invTrainFileName[0])
            y_file = x_file.replace("X", "Y")
            z_file = './datSet/ZDT/ZDT123/{1}/imgZ_{0}.bmp'.format(
                invName, invTrainFileName[1])
        else:
            x_file = './datSet/ZDT/{0}/{1}/imgX_{0}.bmp'.format(
                invName, invTrainFileName[0])
            y_file = x_file.replace("X", "Y")
            z_file = './datSet/ZDT/{0}/{1}/imgZ_{0}.bmp'.format(
                invName, invTrainFileName[1])
    else:
        x_file = './datSet/{0}/{1}/imgX_{0}.bmp'.format(invName,
                                                        invTrainFileName[0])
        y_file = x_file.replace("X", "Y")
        z_file = './datSet/{0}/{1}/imgZ_{0}.bmp'.format(invName,
                                                        invTrainFileName[1])

    return x_file, y_file, z_file
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
    invPathDirMidFeature = r'D:\workspace\DLMOP\CEDN\datFeatureAnalysis' + \
                           r'\midLayerActivate'
    model_integrated_file, model_separated_file = invModelFile
    weight_integrated_file, weight_separated_file = invModelFile
    x_file, y_file, z_file = funSourceFile_subTest(invName, invTrainFileName)
    [model_integrated_path, model_separated_path] , \
                           [weight_integrated_path, weight_separated_path] \
                           = funModelWeightFile_subTest(
                               invModelFile, invWeightFile)

    model_integrated = funLoadModel(model_integrated_path,
                                    weight_integrated_path)
    print('\n=====   {0}   =====   Integrated Model   =====\n'.format(invName))
    parFigPath = os.path.join(invPathDirMidFeature,
                              invName + '_Integrated')
    os.makedirs(parFigPath, exist_ok=True)
    funTestPlot(x_file, z_file,
                invImageSize, model_integrated, parFigPath, invThreshold[0])
    del model_integrated

    model_separated_step1 = funLoadModel(model_separated_path[0],
                                         weight_separated_path[0])
    print('\n=====   {0}   =====   Separated  Model_Step1   =====\n'.
          format(invName))
    parFigPath = os.path.join(invPathDirMidFeature,
                              invName + '_Separated_S1')
    os.makedirs(parFigPath, exist_ok=True)
    funTestPlot(x_file, y_file, invImageSize, model_separated_step1,
                parFigPath, 0.95, parS1Flag=True)
    del model_separated_step1

    model_separated_step2 = funLoadModel(model_separated_path[1],
                                         weight_separated_path[1])
    print('\n=====   {0}   =====   Separated  Model_Step2   =====\n'.
          format(invName))
    parFigPath = os.path.join(invPathDirMidFeature,
                              invName + '_Separated_S2')
    os.makedirs(parFigPath, exist_ok=True)
    funTestPlot(y_file, z_file,
                invImageSize, model_separated_step2, parFigPath, invThreshold[1])
    del model_separated_step2

    return 0

"""==========  MAIN  ====================================================="""
for arg in varArgList:
    subTest(arg)
