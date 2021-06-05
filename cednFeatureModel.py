import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from keras import models
from keras.models import load_model
from keras import backend as K

valName = 'ZDT123'
valImageSize = (2**7, 2**7)

CEDN = load_model('.\datModels\Model_CEDN_Intergrant_ZDT123_128.h5')
#CEDN.summary()
CEDN.load_weights('.\datModels\Weight_Intergrant_ZDT123_128_datX1d.h5')

from vis.utils import utils
from vis.visualization import visualize_activation
import matplotlib.pyplot as plt

layers_names = [layer.name for layer in CEDN.layers]
layer_index = 13
CEDN.layers[len(layers_names) - 1].activation = keras.activations.linear
def subFilterPlot(layer_index=0, filter_indices=0, tv_weight=.1, lp_norm_weight=0.):
    filter_img = visualize_activation(
        CEDN, layer_index, filter_indices=filter_indices,
        input_range=(0.,1.), max_iter=100, verbose=True,
        tv_weight=tv_weight, lp_norm_weight=lp_norm_weight)
    filter_img = np.reshape(filter_img, valImageSize)
    filter_img *= 255.
    img = np.reshape(filter_img, valImageSize)
    plt.imshow(filter_img, cmap='Greys_r')
    plt.title(layers_names[layer_index])
    #plt.figure()
    plt.show()
    plt.close()
    return 0

"""======================================================================="""
'''
layers_count = len(CEDN.layers) 
layers_info = []#[(name, feature_num, feature_size),...,...]

for index, layer in enumerate(CEDN.layers[: layers_count]):
    name = layer.name
    feature_num = layer.output.shape[-1]
    feature_size = layer.output.shape[1]
    layers_info.append((name, feature_num, feature_size, index))
layer_index = utils.find_layer_idx(CEDN, layers_info[-1][0])
CEDN.layers[layer_index].activation = keras.activations.linear

margin = 5
size = valImageSize[0]
images_per_row = 16
scale = 0.5 / size

pick_index = 0
layers_infor = layers_info[pick_index]
for layer_info in layers_info:
    layer_name, feature_num, feature_size, layer_index = layer_info
    cols_num = feature_num // images_per_row
    results = np.zeros((size * cols_num + margin * (cols_num - 1),
                        size * images_per_row + margin * (images_per_row - 1),
                        1))

    for col in range(cols_num):
        for row in range(images_per_row):
            filter_img = visualize_activation(
                CEDN, layer_index, filter_indices=col*images_per_row + row,
                input_range=(0.,1.), max_iter=40, verbose=False)
            filter_img = np.reshape(filter_img, valImageSize + (1,))
            filter_img *= 255.
            horizontal_start = col * size + col * margin
            horizontal_end = horizontal_start + size
            vertical_start = row * size + row * margin
            vertical_end = vertical_start + size
            results[horizontal_start : horizontal_end,
                    vertical_start : vertical_end, :] = filter_img
    plt.figure(figsize=(scale * results.shape[1],
                        scale * results.shape[0]))
    plt.title(layer_name)
    results = np.reshape(results, (np.shape(results)[0], np.shape(results)[1]))
    plt.imshow(results, aspect='auto', cmap='Greys')
    plt.show()
plt.close()
'''
