# xây dựng index cho các feature vector


import pickle

import faiss 

import time 


with open ('pickle/feature.pkl', 'rb') as f:
    features  = pickle.load(f)

M = 32
d = features.shape[1]
start = time.time()
index = faiss.IndexHNSWFlat(d, M)   # build the index
#index.train(xb)
#print(index.is_trained)
index.add(features)                  # add vectors to the index
print ("Indexing time = ", time.time() - start)
faiss.write_index(index, 'index/HNSWFlat2.index') 


print(features.shape)

index = faiss.read_index('index/HNSWFlat2.index')

print("lien")