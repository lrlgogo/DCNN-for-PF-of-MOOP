import keras
import keras.backend as K
import numpy as np
from keras import Model
from keras import models
from keras import layers
import random
import matplotlib.pyplot as plt
"""===   Global Parameters   ============================================="""
valImageSize = (2**7, 2**7)
valNumDat = 1*5120#2048 * 5 * 1#2 * 4096
valDatDir = './datSet/ZDT/ZDT123'#'./datSet/ZDT4'
valTrainEpochs = 100
valBatchSize = 32
valWeightFile = 'VAE_Weight_{}.h5'.format('ZDT123_S')
valLatentDim = 5
valKLLossCoefficient = -5e-1
"""===   Data Preprocessing   ============================================"""
from keras.preprocessing import image
import os
import time

varTrainXDir = os.path.join(valDatDir, 'datX1')
varTrainYDir = os.path.join(valDatDir, 'datY1')
varTrainX = []
varTrainY = []
print('Loading...')
timStart = time.time()
for i in range(valNumDat):
    Xfname = os.path.join(varTrainXDir, 'imgX_{}.bmp'.format(i))
    Yfname = os.path.join(varTrainYDir, 'imgY_{}.bmp'.format(i))
    imgX = image.load_img(Xfname, color_mode='grayscale')
    imgY = image.load_img(Yfname, color_mode='grayscale')

    imgX = image.img_to_array(imgX)
    imgX = list(imgX)
    varTrainX.append(imgX)

    imgY = image.img_to_array(imgY)
    imgY = list(imgY)
    varTrainY.append(imgY)
timEnd = time.time()
print('Loading Completed. Used %ds.' % (timEnd - timStart))
print('Data Separating and Preprocessing...')
timStart = time.time()
varPar1, varPar2 = [1*3200, 1*4160]
varTrainDatX = varTrainX[: varPar1]
varValidDatX = varTrainX[varPar1 : varPar2]
varTestDatX = varTrainX[varPar2 : valNumDat]
varTrainDatY = varTrainY[: varPar1]
varValidDatY = varTrainY[varPar1 : varPar2]
varTestDatY = varTrainY[varPar2 : valNumDat]

varTrainDatX = np.asarray(varTrainDatX)
varValidDatX = np.asarray(varValidDatX)
varTestDatX = np.asarray(varTestDatX)
varTrainDatY = np.asarray(varTrainDatY)
varValidDatY = np.asarray(varValidDatY)
varTestDatY = np.asarray(varTestDatY)

varTrainDatX = varTrainDatX.astype('uint8') / 255
varValidDatX = varValidDatX.astype('uint8') / 255
varTestDatX = varTestDatX.astype('uint8') / 255
varTrainDatY = varTrainDatY.astype('uint8') / 255
varValidDatY = varValidDatY.astype('uint8') / 255
varTestDatY = varTestDatY.astype('uint8') / 255
timEnd = time.time()
print('Separating and Preprocessing Complete. Used %ds' % (timEnd - timStart))
"""===   Net Models   ===================================================="""
valHeight, valWidth = valImageSize
#  Encoder Net
input_img = keras.Input(shape=(valHeight, valWidth, 1))
x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, padding='same', activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

shape_before_flatten = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

z_mean = layers.Dense(valLatentDim)(x)
z_log_var = layers.Dense(valLatentDim)(x)

#  Sampling Layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], valLatentDim),
                              mean=0., stddev=1.)
    return z_mean + epsilon * K.exp(0.5 * z_log_var)
z = layers.Lambda(sampling, output_shape=(valLatentDim,))([z_mean, z_log_var])

#  Decoder Net
decoder_input = layers.Input(K.int_shape(z)[1:])

x = layers.Dense(np.prod(shape_before_flatten[1:]),
                 activation='relu')(decoder_input)
x = layers.Reshape(shape_before_flatten[1:])(x)
x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu',
                           strides=(2, 2))(x)
x = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
decoder = Model(decoder_input, x)

z_decoded = decoder(z)
"""
#  VAE Custom Loss Layer
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, y, z_decoded):
        y = K.flatten(y)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(y, z_decoded)
        kl_loss = valKLLossCoefficient * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        y = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(y, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return y

y = CustomVariationalLayer()([input_img])
"""
#  VAE Custom Loss Function
def vaeLoss(y_true, y_pred):
    y = K.flatten(y_true)
    z_decoded = K.flatten(y_pred)
    xent_loss = keras.metrics.binary_crossentropy(y, z_decoded)
    kl_loss = valKLLossCoefficient * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

#  VAE Model
VAE = Model(input_img, z_decoded)
VAE.summary()
"""===   Train Setting   ================================================="""
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='acc', patience=20),
    keras.callbacks.ModelCheckpoint(filepath=valWeightFile,
                                    monitor='val_loss',
                                    save_best_only=True,),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=20)
    ]
VAE.compile(optimizer='rmsprop',#'adam',#keras.optimizers.RMSprop(lr=1e-4),
             loss=vaeLoss,#'binary_crossentropy',
             metrics=['acc'])
history = VAE.fit(varTrainDatX, varTrainDatY,
                   batch_size=valBatchSize, epochs=valTrainEpochs, verbose=2,
                   callbacks=callbacks_list,
                   validation_data=[varValidDatX, varValidDatY])
VAE.save_weights('Weight_Temp_VAE.h5')
VAE.save('Model_VAE.h5')
"""===   Train Plot   ===================================================="""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
plt.close()

"""===   Test Plot   ====================================================="""
#VAE.load_weights(valWeightFile)
print('Test on test set({} elements):[loss, accuracy]'.format(
    len(varTestDatX)))
print(4 * ' ',VAE.evaluate(varTestDatX, varTestDatY, verbose=0))

def funOtherTest(Xfname, Yfname):
    imgX = image.load_img(Xfname, color_mode='grayscale')
    imgY = image.load_img(Yfname, color_mode='grayscale')
    imgX = image.img_to_array(imgX)
    imgY = image.img_to_array(imgY)
    imgY = imgY.reshape(valImageSize)
    imgX = imgX.astype('float32') / 255
    imgX = imgX.reshape((1,) + np.shape(imgX))
    imgYP = VAE.predict(imgX, verbose=0)
    imgYP = imgYP.reshape(valImageSize)
    imgX = np.asarray(imgX, dtype='float32')
    imgX *= 255
    imgX = imgX.reshape(valImageSize)
    imgX = np.clip(imgX, 0, 255).astype('float32')
    plt.imshow(imgX, cmap='Greys_r')
    plt.figure()

    imgYP = np.asarray(imgYP, dtype='float32')
    imgYP *= 255
    imgYP = np.clip(imgYP, 0, 255).astype('float32')
    plt.imshow(imgYP, cmap='Greys_r')
    plt.figure()

    plt.imshow(imgY, cmap='Greys_r')
    plt.show()
    plt.close()

def funTestPlot(parIndex=None,
                parThreshold=0.7):
    if parIndex == None:
        random.seed()
        parIndex = random.randint(0, len(varTestDatX))
    imgX = varTestDatX[parIndex]
    imgX = imgX.reshape((1,) + np.shape(imgX))
    imgY = VAE.predict(imgX, verbose=0)
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
                if imgY[i][j] > 0.9:
                    imgY[i][j] = 1.01
                else:
                    imgY[i][j] = -0.01
                varIndex = i
            else:
                imgY[i][j] = 1.01
    
    imgTemp = varTestDatY[parIndex]
    imgTemp = imgTemp.reshape(valImageSize)

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
    plt.plot(x1, y1, 'r.', ms=2, label='Predict')
    plt.plot(x2, y2, '+', ms=4, label='Theorize')
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
    plt.show()
    plt.close()
    
funTestPlot(random.randint(0, len(varTestDatX)))
