import _pickle as cPickle
import numpy as np
import time
import scipy.ndimage
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.layers.pooling as pool
import keras.models as models
import keras.utils.np_utils as kutils
import theano
from keras import backend as K
#from nolearn.lasagne import visualize
#import matplotlib.pyplot as plt

def CifarTrainingImages(path):
	i=1
	data=np.array([])
	labels=np.array([])
	while i<=5:
		fo = open(path+'/data_batch_'+str(i), 'rb')
		dict = cPickle.load(fo,encoding='bytes')
		if i==1:
			data=dict[b"data"]
			labels=dict[b"labels"]
		else:
			data=np.concatenate((data, dict[b"data"]), axis=0)
			labels=np.concatenate((labels, dict[b"labels"]), axis=0)
		fo.close()
		i=i+1
	data=data.reshape((50000,3,32,32))
	print('reshaping to 244x244')
	data=scipy.ndimage.zoom(data, (1,1, 7, 7))
	return data,labels

def CifarTestImages(path):
	i=1
	data=np.array([])
	labels=np.array([])
	fo = open(path+'/test_batch'+str(i), 'rb')
	dict = cPickle.load(fo,encoding='bytes')
	data=dict[b"data"]
	labels=dict[b"labels"]
	fo.close()
	data=data.reshape((10000,3,32,32))
	print('reshaping to 244x244')
	data=scipy.ndimage.zoom(data, (1,1, 7, 7))
	return data,labels



trainX,trainY=CifarTrainingImages("../../Cifar10")
trainX=trainX.astype(float)
trainX/=255.0


NoClasses=10

NoEpoch=5

# K.set_image_dim_ordering('th')

tic=time.time()
cnn=models.Sequential()

conv1_1=conv.Convolution2D(64,3,3,border_mode='same',input_shape=(3,224,224))
cnn.add(conv1_1)
relu1_1 = core.Activation('relu')
cnn.add(relu1_1)

conv1_2=conv.Convolution2D(64,3,3,border_mode='same',input_shape=(3,224,224))
cnn.add(conv1_2)
relu1_2 = core.Activation('relu')
cnn.add(relu1_2)


cnn.add(pool.MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid",dim_ordering='default'))


conv2_1=conv.Convolution2D(128,3,3,border_mode='same')
cnn.add(conv2_1)
relu2_1 = core.Activation('relu')
cnn.add(relu2_1)

conv2_2=conv.Convolution2D(128,3,3,border_mode='same')
cnn.add(conv2_2)
relu2_2 = core.Activation('relu')
cnn.add(relu2_2)


cnn.add(pool.MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid",dim_ordering='default'))


conv3_1=conv.Convolution2D(256,3,3,border_mode='same')
cnn.add(conv3_1)
relu3_1 = core.Activation('relu')
cnn.add(relu3_1)

conv3_2=conv.Convolution2D(256,3,3,border_mode='same')
cnn.add(conv3_2)
relu3_2 = core.Activation('relu')
cnn.add(relu3_2)


conv3_3=conv.Convolution2D(256,3,3,border_mode='same')
cnn.add(conv3_3)
relu3_3 = core.Activation('relu')
cnn.add(relu3_3)

conv3_4=conv.Convolution2D(256,3,3,border_mode='same')
cnn.add(conv3_4)
relu3_4 = core.Activation('relu')
cnn.add(relu3_4)


cnn.add(pool.MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid",dim_ordering='default'))

conv4_1=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv4_1)
relu4_1 = core.Activation('relu')
cnn.add(relu4_1)

conv4_2=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv4_2)
relu4_2 = core.Activation('relu')
cnn.add(relu4_2)


conv4_3=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv4_3)
relu4_3 = core.Activation('relu')
cnn.add(relu4_3)

conv4_4=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv4_4)
relu4_4 = core.Activation('relu')
cnn.add(relu4_4)


cnn.add(pool.MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid",dim_ordering='default'))

conv5_1=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv5_1)
relu5_1 = core.Activation('relu')
cnn.add(relu5_1)

conv5_2=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv5_2)
relu5_2 = core.Activation('relu')
cnn.add(relu5_2)
"time taken = "+str(toc-tic)+"sec"

conv5_3=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv5_3)
relu5_3 = core.Activation('relu')
cnn.add(relu5_3)

conv5_4=conv.Convolution2D(512,3,3,border_mode='same')
cnn.add(conv5_4)
relu5_4 = core.Activation('relu')
cnn.add(relu5_4)


cnn.add(pool.MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode="valid",dim_ordering='default'))

cnn.add(core.Flatten())

cnn.add(core.Dropout(0.4))

cnn.add(core.Dense(4096,activation="relu"))

cnn.add(core.Dropout(0.4))

cnn.add(core.Dense(4096,activation="relu"))

cnn.add(core.Dropout(0.4))

cnn.add(core.Dense(1000,activation="relu"))

cnn.add(core.Dropout(0.4))

cnn.add(core.Dense(10,activation="softmax"))

cnn.summary()

cnn.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

cnn.fit(trainX, trainY, batch_size=256, nb_epoch=NoEpoch, verbose=1)
toc=time.time()

print("time taken = "+str(toc-tic)+"sec")


testX,testY = CifarTestImages("../../Cifar10")
testX = testX.astype(float)
testX /= 255.0

# Getting predictions for test data from the trained model
yPred = cnn.predict_classes(testX)
# np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
c=0
i=0
while i<len(yPred):
	if yPred[i]==testY[i]:
		c=c+1
	i=i+1

acc=c/len(testY)

print("Accuracy is"+str(acc))
