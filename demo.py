print("liên")

import gensim
import keras
import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Activation, Lambda, Flatten, concatenate, Reshape
from keras.models import Model
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# from PIL import Image
import cv2
# from google.colab.patches import cv2_imshow
from keras.layers.normalization import BatchNormalization

import faiss

print("liên")

from keras.models import Sequential


class GetDataset:
  
    
  def get_img(self, data_path):
    
    img_size = 256
    im = cv2.imread(data_path)
    im_resized = cv2.resize(im, (img_size,img_size), interpolation=cv2.INTER_LINEAR)

    return im_resized

  
  def get_dataset(self, dataset_path = '/content/drive/My Drive/20191/Hệ CSDL đa phương tiện/Data'):
    
    X = []
    Y = []
    
    i = 0 
    
    for label in os.listdir(dataset_path):
        print(label)
       
        data_path = dataset_path + '/' + label
        
        
        for file in os.listdir(data_path):
            
            filename = data_path + '/' + file
         
            img = self.get_img(filename)
            X.append(img)
            Y.append(i)
          
        i = i + 1
        
        
     # Create dateset:
    X = np.array(X).astype('float32')/255.
    Y = np.array(Y).astype('float32')
    #Y = to_categorical(Y, 2)
    
    print(X.shape)
    print(Y.shape)
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test
    

dt = GetDataset()

X_train, X_test, Y_train, Y_test = dt.get_dataset('data')



# About Dataset:
img_size = X_train.shape[1] 
print('Training shape:', X_train.shape)
print(X_train.shape[0], 'sample,',X_train.shape[1] ,'x',X_train.shape[2] ,'size RGB image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,',X_test.shape[1] ,'x',X_test.shape[2] ,'size RGB image.\n')

print('Examples:')
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # Display some data:
    ax = plt.subplot(1, n, i)
    plt.imshow(X_train[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#Convolutional Autoencoder
input_img = Input(shape=(256, 256, 3))
def Model_CNN(input_img):
  #encoding
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)

  #decoding
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

  autoencoder = Model(input_img, decoded)
  autoencoder.compile(optimizer='adam', loss='mse')

  autoencoder.summary()

  return autoencoder, encoded


autoencoder,encoded = Model_CNN(input_img)
print(autoencoder.summary())

history2 = autoencoder.fit(X_train, X_train,
                epochs=2,
                batch_size=256,
                shuffle=True,
                validation_split=0.1,
                verbose = 1)

#plot our loss 
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# use our encoded layer to encode the training input
encoder = Model(input_img, encoded)
#encoded_input = Input(shape=(encoding_dim,))
