from flask import Flask, render_template, request, url_for, redirect
import cv2 
import time
import faiss
import numpy as np 
import matplotlib.pyplot as plt 
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from IPython.display import Image



app = Flask(__name__)

# load model
# model = load_model('model/model.h5')

def retrieval(query_path = 'query/query.jpg'):

    print('-------Tim kiem---------')
    img_size = 256
    im = cv2.imread(query_path)
    query_img = cv2.resize(im, (img_size,img_size), interpolation=cv2.INTER_LINEAR)

    print(query_img.shape)


    X= []
    X.append(query_img)
    X = np.array(X).astype('float32')/255.
    print(X.shape)
    model = load_model('model/model.h5')
    encoded_data = model.predict(X)
    print("lien")

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
            path = 'data' + '/'+ str(folder_cur)  +'/' + 'img_0000000' + str(file) + '.jpg'
        else:
            path = 'data' + '/' +  str(folder_cur) +'/' + 'img_000000' + str(file) + '.jpg'

        print(path)

        folder.append(path)

    return folder 


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST','GET'])

def result():

    print('Ket  qua')
    if request.method == 'POST':
        print('-------------1---------------')
        if request.form.get('search_btn') == 'search_btn':

            print('---------2---------')
            
            folder = retrieval(query_path = 'query/query.jpg')
            path_result = "/home/builien2010/csdl/Code/"+folder[0]

            # folder = ['data/img_00000001.jpg']
            print(folder)

            report_template = render_template('result.html',folder=path_result)

            return report_template


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
