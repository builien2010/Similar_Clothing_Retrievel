import cv2 
import time
import faiss
import numpy as np 
import matplotlib.pyplot as plt 
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from IPython.display import Image, display


# load model
model = load_model('model/model.h5')




def retrievel(query_path = 'query/query.jpg'):
    img_size = 256
    im = cv2.imread(query_path)
    query_img = cv2.resize(im, (img_size,img_size), interpolation=cv2.INTER_LINEAR)

    print(query_img.shape)


    X= []
    X.append(query_img)
    X = np.array(X).astype('float32')/255.
    encoded_data = model.predict(X)

    print("encoded_data.shape :", encoded_data.shape)
    xq = encoded_data.reshape((-1, np.prod(encoded_data.shape[1]*encoded_data.shape[2]*encoded_data.shape[3])))

    print("xq: ", xq)

    print(xq.shape)

    index = faiss.read_index('index/HNSWFlat2.index')

    k = 4

    IVFPQ_D, IVFPQ_I = index.search(xq, k) 


    folder = []

    for i, img_index in enumerate(IVFPQ_I[0]):
        print ('{}. '.format(i+1))
        
        print(img_index)

        folder_cur = int(img_index/45) + 1

        print(folder_cur)

        file = img_index - 45*(folder_cur - 1)

        print(file)

        if ( file < 10):
            path = 'data' + '/' + 'img_0000000' + str(file) + '.jpg'
        else:
            path = 'data' + '/' + 'img_000000' + str(file) + '.jpg'

        print(path)

        folder.append(path)

        

    return folder 


folder = retrievel()

print(folder)





    
    
