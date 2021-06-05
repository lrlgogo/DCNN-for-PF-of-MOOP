import keras
import numpy as np
from keras import models
from keras import layers
import random
import matplotlib.pyplot as plt
"""===   Global Parameters   ============================================="""
valImageSize = (2**7, 2**7)
valNumDat = 1*5120#2048 * 5 * 1#2 * 4096
valDatDir = './datSet/ZDT/ZDT123'#'./datSet/ZDT4'
valTrainEpochs = 5
valBatchSize = 16#32
valWeightFile = 'Weight_{}.h5'.format('ZDT123Func')
#GAN Train
valIteration = 16000
"""===   Data Preprocessing   ============================================"""
from keras.preprocessing import image
import os
import time

varTrainXDir = os.path.join(valDatDir, 'datX')
varTrainYDir = os.path.join(valDatDir, 'datY')
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
#"""
def funGeneratorCEDN():
    generator_input = keras.Input(shape=(valHeight, valWidth, 1))
    
    x = layers.Conv2D(32, 3)(generator_input)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, 3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, 3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, 3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    #Reverse
    x = layers.Conv2DTranspose(256, 1)(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(256, 3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(256, 3)(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(128, 3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(128, 3)(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, 3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 3)(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, 3)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(32, 3)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(1, 5, activation='sigmoid')(x)
    generator = keras.models.Model(generator_input, x)

    return generator

def funDiscriminatorNet():
    discriminator_input = layers.Input(shape=(valHeight, valWidth, 1))

    x = layers.Conv2D(256, 4, strides=2)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    discriminator = keras.models.Model(discriminator_input, x)
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=8e-4,
        clipvalue=1.0,
        decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')

    return discriminator
'''Adversarial Net'''
CEDN = funGeneratorCEDN()
DNet = funDiscriminatorNet()
CEDN.load_weights('Weight_ZDT123.h5')
DNet.trainable = False

GAN_input = keras.Input(shape=(valHeight, valWidth, 1))
GAN_output = DNet(CEDN(GAN_input))
GAN = keras.models.Model(GAN_input, GAN_output)
GAN_optimizer = keras.optimizers.RMSprop(
    lr=5e-5,
    clipvalue=1.0,
    decay=1e-8)
GAN.compile(optimizer=GAN_optimizer,
            loss='binary_crossentropy')
#"""
CEDN.summary()
DNet.summary()
GAN.summary()
"""===   Train Setting   ================================================="""
#GAN Train
varSaveDir = r'OUT/'
varStart = 0
for step in range(1, valIteration):
    varTimeStart = time.time()
    varStop = varStart + valBatchSize
    varInputX = varTrainDatX[varStart : varStop]
    varInputY = varTrainDatY[varStart : varStop]
    varOutputY = CEDN.predict(varInputX)
    varCombinedY = np.concatenate([varOutputY, varInputY])
    varDNetLabel = np.concatenate([0.95 * np.ones((valBatchSize, 1)),
                                   np.zeros((valBatchSize, 1))])
    varDNetLabel += 0.5 * np.random.random(varDNetLabel.shape)

    for i in range(5):
        varDNetLoss = DNet.train_on_batch(varCombinedY, varDNetLabel)

    #varGANInputX = varTrainDatX[varStart]
    #varGANInputX = np.reshape(varGANInputX, (1,) + np.shape(varGANInputX))
    varGANInputY = np.zeros((valBatchSize, 1))

    varGANLoss = GAN.train_on_batch(varInputX, varGANInputY)
    
    varStart += valBatchSize
    if varStart >= len(varTrainDatX):
        varStart = 0
        
    if step % 10 == 0:
        GAN.save_weights('gan.h5')
        print(15 * '=' + 'steps:{}'.format(step) + 15 * '=')
        print('discriminator loss:', varDNetLoss)
        print('adversarial   loss:', varGANLoss)
        print('sum    of     loss:', varDNetLoss + varGANLoss)
        random.seed()
        varOutputIndex = random.randint(0, valBatchSize - 1)
        img = image.array_to_img(
            np.concatenate([varOutputY[varOutputIndex],
                            varInputY[varOutputIndex]]) * 255., scale=False)
        img.save(os.path.join(varSaveDir, str(step) + '_' +
                              str(varOutputIndex) + '.png'))
        varTimeEnd = time.time()
        print('time of stpes used:{}s'.format(varTimeEnd - varTimeStart))

"""CEDN Train
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='acc', patience=10),
    keras.callbacks.ModelCheckpoint(filepath=valWeightFile,
                                    monitor='val_loss',
                                    save_best_only=True,),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=10)
    ]
CEDN.compile(optimizer='adam',#'rmsprop',#keras.optimizers.RMSprop(lr=1e-4),
             loss='binary_crossentropy',
             metrics=['acc'])
history = CEDN.fit(varTrainDatX, varTrainDatY,
                   batch_size=valBatchSize, epochs=valTrainEpochs, verbose=2,
                   callbacks=callbacks_list,
                   validation_data=[varValidDatX, varValidDatY])
CEDN.save_weights('Weight_TempFunc.h5')
CEDN.save('Model_CEDN_2_Func.h5')
"""
"""===   Train Plot   ===================================================="""
"""CEDN Plot
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
"""
"""===   Test Plot   ====================================================="""
#CEDN.load_weights(valWeightFile)
"""CEDN Test
print('Test on test set({} elements):[loss, accuracy]'.format(
    len(varTestDatX)))
print(4 * ' ',CEDN.evaluate(varTestDatX, varTestDatY, verbose=0))
"""
def funOtherTest(Xfname, Yfname):
    imgX = image.load_img(Xfname, color_mode='grayscale')
    imgY = image.load_img(Yfname, color_mode='grayscale')
    imgX = image.img_to_array(imgX)
    imgY = image.img_to_array(imgY)
    imgY = imgY.reshape(valImageSize)
    imgX = imgX.astype('float32') / 255
    imgX = imgX.reshape((1,) + np.shape(imgX))
    imgYP = CEDN.predict(imgX, verbose=0)
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
