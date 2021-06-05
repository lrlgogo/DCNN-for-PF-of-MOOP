import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.models import load_model
from keras import backend as K

valImageSize = (2**7, 2**7)
valBatchSize = 32

CEDN = load_model('.\datModels\Model_CEDN_Intergrant_ZDT123_128.h5')
CEDN.load_weights('.\datModels\Weight_Intergrant_ZDT123_128_datX1d.h5')

valInput_x = np.ones((valBatchSize,) + valImageSize + (1,))
valInput_y = np.ones((valBatchSize,) + valImageSize + (1,))


model_input = models.Sequential(name='input')
model_input.add(layers.LocallyConnected2D(
    1, 1, use_bias=False, input_shape=(valImageSize + (1,)), name='outLayer'))

model = models.Sequential()
model.add(model_input)
model.add(CEDN)

CEDN.trainable = False
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(valInput_x, valInput_y, batch_size=valBatchSize, epochs=100, verbose=2)

