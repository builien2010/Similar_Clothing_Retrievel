import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.



import time
import faiss 

print("lien")

# Indexing using HNSW

M = 32
start = time.time()
index = faiss.IndexHNSWFlat(d, M)   #  # build the index
#index.train(xb)
#print(index.is_trained)
index.add(xb)                  # add vectors to the index
print ("Indexing time = ", time.time() - start)