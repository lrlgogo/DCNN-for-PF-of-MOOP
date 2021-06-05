import keras
import numpy as np
from keras import models
from keras import layers
import random
import matplotlib.pyplot as plt
"""===   Global Parameters   ============================================="""
valName = 'POL'
valTrainFileName = ['datXd2', 'datYd2']
valImageSize = (2**7, 2**7)
valNumDat = 1*1024#1*5120#2048 * 5 * 1#2 * 4096
valDatDir = './datSet/{}'.format(valName)#ZDT/ZDT4'#'./datSet/ZDT4'
valTrainEpochs = 50
valBatchSize = 8
valWeightFile = './datModels/Weight_Separation_S1_{0}_{1}_{2}.h5'.format(
    valName, valImageSize[0], valTrainFileName[0])
"""===   Data Preprocessing   ============================================"""
from keras.preprocessing import image
import os
import time

varTrainXDir = os.path.join(valDatDir, valTrainFileName[0])
varTrainYDir = os.path.join(valDatDir, valTrainFileName[1])
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
# varPar1, varPar2 = [1*3200, 1*4160]
varPar1, varPar2 = [512, 768]
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
CEDN = models.Sequential()

CEDN.add(layers.Conv2D(32, 1, activation='relu',
                       input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(32, 3, activation='relu'))#,
                       #input_shape=(valWidth, valHeight, 1)))
CEDN.add(layers.Conv2D(32, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2D(64, 1, activation='relu'))
CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.Conv2D(64, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2D(128, 1, activation='relu'))
CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.Conv2D(128, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))

CEDN.add(layers.Conv2D(256, 1, activation='relu'))
CEDN.add(layers.Conv2D(256, 3, activation='relu'))
CEDN.add(layers.Conv2D(256, 3, activation='relu'))
CEDN.add(layers.MaxPooling2D((2, 2)))
#
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
CEDN.summary()
"""===   Train Setting   ================================================="""
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='acc', patience=40),
    keras.callbacks.ModelCheckpoint(filepath=valWeightFile,
                                    monitor='val_loss',
                                    save_best_only=True,),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.5,
                                      patience=10)
    ]

CEDN.compile(optimizer='adam',#'rmsprop',#keras.optimizers.RMSprop(lr=1e-4),
             loss='binary_crossentropy',
             metrics=['acc'])
history = CEDN.fit(varTrainDatX, varTrainDatY,
                   batch_size=valBatchSize, epochs=valTrainEpochs, verbose=2,
                   callbacks=callbacks_list,
                   validation_data=[varValidDatX, varValidDatY])
CEDN.save_weights('./datModels/Weight_Temp_Separation_S1_{0}_{1}_{2}.h5'.format(
    valName, valImageSize[0], valTrainFileName[0]))
CEDN.save('./datModels/Model_CEDN_Separation_S1_{0}_{1}.h5'.format(
    valName, valImageSize[0]))
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
#CEDN.load_weights(valWeightFile)
print('Test on test set({} elements):[loss, accuracy]'.format(
    len(varTestDatX)))
print(4 * ' ',CEDN.evaluate(varTestDatX, varTestDatY, verbose=0))

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

def funTestPlot(parThreshold=0.95,
                parIndex=None):
    if parIndex == None:
        random.seed()
        parIndex = random.randint(0, len(varTestDatX) - 1)
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
    
funTestPlot(random.randint(0, len(varTestDatX) - 1))
