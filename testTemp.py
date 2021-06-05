import keras
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from keras import models
from keras.models import load_model
from keras import backend as K

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
"""======================================================================="""
def funLoadModel(parModelFile, parWeightFile):
    model = models.load_model(parModelFile)
    model.load_weights(parWeightFile)
    #model.summary()

    return model

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

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-9)
    x *= 0.3

    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(model, layer_name, filter_index, size, iteration_num):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-9)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 1)) * 0.2 + 0.5

    step = 1.
    for i in range(iteration_num):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img), loss_value, grads_value

def subSaveFilter(args):
    invBaseDir, model, invIteration, invImageSize, invFilterNum = args

    img_per_row = 4
    margin = 5
    size = invImageSize[0]
    scale = 1. / size

    for index, layer in enumerate(model.layers):
        rows_num = invFilterNum // img_per_row
        if layer == model.layers[-1]:
            rows_num = 1
            img_per_row = 1
        results = np.zeros(
            (size * img_per_row + margin * (img_per_row - 1),
             size * rows_num + margin * (rows_num - 1)))
        invImgPath = os.path.join(invBaseDir, '{}_{}.png'.format(index,
                                                                 layer.name))

        for row in range(rows_num):
            for col in range(img_per_row):
                filter_img, _, _ = generate_pattern(
                    model, layer.name, size=size, iteration_num = invIteration,
                    filter_index=row * img_per_row + col)
                filter_img = np.reshape(filter_img, invImageSize)
                horizontal_start = col * (size + margin)
                horizontal_end = horizontal_start + size
                vertical_start = row * (size + margin)
                vertical_end = vertical_start + size
                results[horizontal_start : horizontal_end,
                        vertical_start : vertical_end] = filter_img
        plt.figure(figsize=(scale * results.shape[1],
                            scale * results.shape[0]))
        plt.title(layer.name)
        plt.imshow(results, cmap='Greys_r')
        plt.axis('off')
        plt.savefig(invImgPath, bbox_inches='tight')
        plt.close()

    return 0

def subMain(args):
    invName, invTrainFileName, invImageSize, \
             invModelFile, invWeightFile, invThreshold = args
    invPathDirMaxAct = r'D:\workspace\DLMOP\CEDN\datFeatureAnalysis' + \
                       r'\maxFilterActivate'
    invPathDirMaxAct = os.path.join(invPathDirMaxAct, invName)
    model_integrated_file, model_separated_file = invModelFile
    weight_integrated_file, weight_separated_file = invModelFile
    [model_integrated_path, model_separated_path] , \
                           [weight_integrated_path, weight_separated_path] \
                           = funModelWeightFile_subTest(
                               invModelFile, invWeightFile)
    model_file = {'model_integrated' : [model_integrated_path,
                                        weight_integrated_path],
                  'model_separated_s1' : [model_separated_path[0],
                                          weight_separated_path[0]],
                  'model_separated_s2' : [model_separated_path[1],
                                          weight_separated_path[1]]
                  }
    for name, value in model_file.items():
        invBaseDir = os.path.join(invPathDirMaxAct, name)
        os.makedirs(invBaseDir, exist_ok=True)
        model = funLoadModel(value[0], value[1])
        args_sub = [invBaseDir, model, 100, invImageSize, 16]
        subSaveFilter(args_sub)

    return 0
"""=====   MAIN   ========================================================"""
'''
for args in varArgList[4:]:
    subMain(args)
'''
