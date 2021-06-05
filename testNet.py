import keras
import numpy as np
from keras import models
from keras import layers

valImageSize = (2**7, 2**7)

valHeight, valWidth = valImageSize
"""
CEDN = models.Sequential()
CEDN.add(layers.Conv2D(32, 3, activation='relu',
                       input_shape=(valWidth, 2 * valHeight, 1)))
CEDN.add(layers.Conv2D(32, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 4)))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
#CEDN.add(layers.Conv2D(128, 5, activation='relu'))
#CEDN.add(layers.Conv2D(128, 5, activation='relu'))
#CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(64, 1, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
#CEDN.add(layers.Conv2DTranspose(64, 5, activation='relu'))
CEDN.add(layers.Conv2DTranspose(32, 5, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(32, 5, activation='relu'))
#CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(1, 5, activation='sigmoid'))
"""
"""
CEDN = models.Sequential()
CEDN.add(layers.Conv2D(32, 5, activation='relu',
                       input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(32, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(128, 5, activation='relu'))
CEDN.add(layers.Conv2D(128, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(128, 1, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(128, 5, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 5, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 5, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(32, 4, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(1, 3, activation='sigmoid'))
"""
"""
CEDN = models.Sequential()
CEDN.add(layers.Conv2D(32, 3, activation='relu',
                       input_shape=(valWidth, 2 * valHeight, 1)))
CEDN.add(layers.Conv2D(32, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 4)))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
#CEDN.add(layers.Conv2D(128, 5, activation='relu'))
#CEDN.add(layers.Conv2D(128, 5, activation='relu'))
#CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(64, 1, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
#CEDN.add(layers.Conv2DTranspose(64, 5, activation='relu'))
CEDN.add(layers.Conv2DTranspose(32, 5, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(32, 5, activation='relu'))
#CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(1, 5, activation='softmax'))
"""
"""
CEDN = models.Sequential()
CEDN.add(layers.Conv2D(32, 3, activation='relu',
                       input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(32, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(256, 3, activation='relu'))
CEDN.add(layers.Conv2D(256, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(256, 1, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(128, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(32, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(1, 5, activation='sigmoid'))
#"""
#"""
CEDN = models.Sequential()
CEDN.add(layers.Conv2D(32, 3, activation='relu',
                       input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(32, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(256, 3, activation='relu'))
CEDN.add(layers.Conv2D(256, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(256, 1, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(128, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(128, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(32, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(32, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(1, 5, activation='sigmoid'))
#"""
"""
CEDN = models.Sequential()
CEDN.add(layers.Conv2D(32, 5, activation='relu',
                       input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(32, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.Conv2D(64, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
CEDN.add(layers.Conv2D(128, 5, activation='relu'))
CEDN.add(layers.Conv2D(128, 5, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Flatten())
CEDN.add(layers.Dense(25, activation='relu'))
CEDN.add(layers.Dense(25, activation='relu'))
CEDN.add(layers.Dense(625, activation='sigmoid'))
CEDN.add(layers.Reshape((25, 25, 1)))

CEDN.add(layers.Conv2DTranspose(128, 1, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(128, 5, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 5, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 5, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(32, 4, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(1, 3, activation='sigmoid'))
"""
"""
CEDN = models.Sequential(name='VGG16Like')
#block_1
CEDN.add(layers.Conv2D(64, 3, activation='relu', padding='same',
                       input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
CEDN.add(layers.MaxPooling2D((2, 2)))
#block_2
CEDN.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2D(128, 3, activation='relu', padding='same'))
CEDN.add(layers.MaxPooling2D((2, 2)))
#block_3
CEDN.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2D(256, 3, activation='relu', padding='same'))
CEDN.add(layers.MaxPooling2D((2, 2)))
#block_4
CEDN.add(layers.Conv2D(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2D(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2D(512, 3, activation='relu', padding='same'))
CEDN.add(layers.MaxPooling2D((2, 2)))
#block_5
CEDN.add(layers.Conv2D(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2D(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2D(512, 3, activation='relu', padding='same'))
CEDN.add(layers.MaxPooling2D((2, 2)))

#block_B_1
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(512, 3, activation='relu', padding='same'))
#block_B_2
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(512, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(512, 3, activation='relu', padding='same'))
#block_B_3
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu', padding='same'))
#block_B_4
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(128, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(128, 3, activation='relu', padding='same'))
#block_B_5
CEDN.add(layers.UpSampling2D((2, 2)))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu', padding='same'))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu', padding='same'))

CEDN.add(layers.Conv2DTranspose(1, 1, activation='sigmoid'))
"""
"""
CEDN = models.Sequential(name='FC_Net')
CEDN.add(layers.Reshape((valWidth * valHeight,),
                        input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Dense(1000, activation='relu'))
CEDN.add(layers.Dense(1000, activation='relu'))
CEDN.add(layers.Dense(1000, activation='relu'))
CEDN.add(layers.Dense(valWidth * valHeight, activation='sigmoid'))
CEDN.add(layers.Reshape((valWidth, valHeight, 1)))
"""
#  Reduce Network
"""
CEDN = models.Sequential()

CEDN.add(layers.Conv2D(32, 3, activation='relu',
                       input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(32, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

#CEDN.add(layers.Conv2D(256, 3, activation='relu'))
#CEDN.add(layers.Conv2D(256, 3, activation='relu'))
#CEDN.add(layers.MaxPooling2D((2, 2)))
########
CEDN.add(layers.Conv2DTranspose(64, 1, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))

#CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu'))
#CEDN.add(layers.Conv2DTranspose(256, 3, activation='relu'))
#CEDN.add(layers.UpSampling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(128, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(128, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(64, 3, activation='relu'))
CEDN.add(layers.UpSampling2D((2, 2)))

CEDN.add(layers.Conv2DTranspose(32, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(32, 3, activation='relu'))
CEDN.add(layers.Conv2DTranspose(1, 5, activation='sigmoid'))
#"""
CEDN.summary()

#CEDN.load_weights('Model_Weight_ZDT123_VGG16.h5')
CEDN.save('Model_CEDN_1.h5')
