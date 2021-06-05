import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from keras import models
from keras.models import load_model
from keras import backend as K

valImageSize = (2**7, 2**7)
"""
CEDN = load_model('.\datModels\Model_CEDN_Intergrant_ZDT123_128.h5')
CEDN.summary()
CEDN.load_weights('.\datModels\Weight_Intergrant_ZDT123_128_datX1d.h5')
"""
CEDN = load_model('.\datModels\Model_CEDN_Separation_S1_NSGA_II_9_test_128.h5')
CEDN.summary()
CEDN.load_weights('.\datModels\Weight_Temp_Separation_S1_NSGA_II_9_test_128_datX.h5')

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-9)
    #x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size, iteration_num):
    layer_output = CEDN.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, CEDN.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-9)
    iterate = K.function([CEDN.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.

    step = 1.
    for i in range(iteration_num):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

layers_count = len(CEDN.layers)
layers_info = []#[(name, feature_num, feature_size),...,...]

for layer in CEDN.layers[: layers_count]:
    name = layer.name
    feature_num = layer.output.shape[-1]
    feature_size = layer.output.shape[1]
    layers_info.append((name, feature_num, feature_size))

margin = 5
size = valImageSize[0]
images_per_row = 16
scale = 0.5 / size

pick_index = 0
layers_infor = layers_info[pick_index]
for layer_info in layers_info:
    layer_name, feature_num, feature_size = layer_info
    cols_num = feature_num // images_per_row
    results = np.zeros((size * cols_num + margin * (cols_num - 1),
                        size * images_per_row + margin * (images_per_row - 1),
                        1))

    for col in range(cols_num):
        for row in range(images_per_row):
            filter_img = generate_pattern(layer_name,
                                          col*images_per_row + row,
                                          size=size,
                                          iteration_num=40)
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
