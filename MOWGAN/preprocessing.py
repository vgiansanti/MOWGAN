# import 
import tensorflow as tf
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import cellrank as cr

import networkx as nx
from sklearn.cluster import AgglomerativeClustering


class preprocessing:
    def __init__(self, adata1, adata2, embedding='umap', n_components=5):
        self.adata1 = adata1
        self.adata2 = adata2
        self.embedding = 'umap'
        self.n_components = 5

    def get_embedding(adata1, adata2, embedding, n_components):
        if embedding=='umap':
            mapper_adata1= umap.UMAP(random_state=9999,n_components=n_components,
                                 low_memory=True,n_epochs=1000).fit(adata1.obsm['X_pca'])
            mapper_adata2 = umap.UMAP(random_state=9999, n_components=n_components,
                                  low_memory=True,n_epochs=1000).fit(adata2.obsm['X_pca'])

            adata1_tr = mapper_adata1.transform(adata1.obsm['X_pca'])
            adata2_tr = mapper_adata2.transform(adata2.obsm['X_pca'])

            adata1.obsm['X_UmapPROVA'] = adata1_tr
            adata2.obsm['X_UmapPROVA'] = adata2_tr
    
        if embedding=='pca':
            pca_adata1 = PCA(n_components=n_components,random_state=9999).fit(adata1.X.todense())
            pca_adata2 = PCA(n_components=n_components,random_state=9999).fit(adata2.X.todense())
        
            adata1_tr = pca_adata1.transform(adata1.X.todense())
            adata2_tr = pca_adata2.transform(adata2.X.todense())

            adata1.obsm['X_pcaPROVA'] = adata1_tr
            adata2.obsm['X_pcaPROVA'] = adata2_tr

            T_adata1 = adata1_tr.min(axis=0)
            T_adata2 = adata2_tr.min(axis=0)

            adata1.uns['params_pca'] = {'translation':T_adata1, 'components':pca_adata1.components_, 'mean':pca_adata1.mean_}   
            adata2.uns['params_pca'] = {'translation':T_adata2, 'components':pca_adata2.components_, 'mean':pca_adata2.mean_}   
        
   #     adata1_tr = adata1_tr - T_adata1
   #     adata2_tr = adata2_tr - T_adata2

    def scaler(adata1,adata2,embedding):
        if embedding=='umap':
            adata1_tr = adata1.obsm['X_UmapPROVA']
            adata2_tr = adata2.obsm['X_UmapPROVA']

            scaler_adata1 = MinMaxScaler()
            scaler_adata2 = MinMaxScaler()

            scaler_adata1.fit(adata1_tr)
            scaler_adata2.fit(adata2_tr)

            adata1_tr = scaler_adata1.transform(adata1_tr)
            adata2_tr = scaler_adata2.transform(adata2_tr)
        
        if embedding=='pca':
            adata1_tr = adata1.obsm['X_pcaPROVA'] - adata1.uns['params_pca']['translation']
            adata2_tr = adata2.obsm['X_pcaPROVA'] - adata2.uns['params_pca']['translation']    
    
        return adata1_tr, adata2_tr
    
    def transition(adata1,adata2):
        ck = cr.tl.kernels.ConnectivityKernel(adata1)
        ck.compute_transition_matrix()
        Tr_adata1 = tf.convert_to_tensor(ck.transition_matrix.A)

        ck = cr.tl.kernels.ConnectivityKernel(adata2)
        ck.compute_transition_matrix()
        Tr_adata2 = tf.convert_to_tensor(ck.transition_matrix.A)
    
        return Tr_adata1, Tr_adata2
    
    def hierarchical(adata1,adata2,embedding):
    
        adata1_tr, adata2_tr = preprocessing.scaler(adata1,adata2,embedding)

        c_1 = AgglomerativeClustering(linkage='ward',n_clusters=15).fit(adata1_tr)
        c_2 = AgglomerativeClustering(linkage='ward',n_clusters=15).fit(adata2_tr)
        
        r1 = pd.DataFrame(adata1_tr)
        r1['cluster'] = c_1.labels_

        r2 = pd.DataFrame(adata2_tr)
        r2['cluster'] = c_2.labels_

        r1.sort_values(by='cluster', inplace=True)
        r2.sort_values(by='cluster', inplace=True)

        r1 = r1.drop(columns=['cluster']).to_numpy()   # questi sono quelli che vanno in input a train_test, sarebbe adata1_tr
        r2 = r2.drop(columns=['cluster']).to_numpy()
    
        return r1,r2
    
    
    def train_test(d1, d2, d1_X, d2_X, embedding, method='random_walk',**kwds):
    
        if method=='random_walk':
            G1,G2 = preprocessing.transition(d1,d2)
            n_start = 1
            p1 = tf.zeros(d1.shape[0], dtype=tf.dtypes.float64)
            p2 = tf.zeros(d2.shape[0], dtype=tf.dtypes.float64)

            idx = tf.random.uniform(shape=[n_start], minval=0, maxval=d1.shape[0],dtype=tf.int64)
    
            p1 = p1.numpy()
            p1[idx] = 1
            p1 = tf.convert_to_tensor(p1)
    
            p2 = p2.numpy()
            p2[idx] = 1
            p2 = tf.convert_to_tensor(p2)
    
            p1 = tf.reshape(p1, shape=(-1,1))
            p2 = tf.reshape(p2, shape=(-1,1))
    
            walk_length = 100
            for k in range(walk_length):
                p1 = tf.reshape(tf.tensordot(G1,p1, axes=1), shape=(-1)) 
                p2 = tf.reshape(tf.tensordot(G2,p2, axes=1), shape=(-1))

            n_batch=1024

            c1 = np.argsort(p1)[-n_batch:]
            c2 = np.argsort(p2)[-n_batch:]
        
            BATCH_SIZE = min(len(c1), len(c2))
            if ( BATCH_SIZE < 1024):
                BATCH_SIZE = BATCH_SIZE
            else: BATCH_SIZE = 1024
    
            TRAIN_BUF=60000
            TEST_BUF=10000
            N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
            N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)

            train_d1 = d1_X[c1[:BATCH_SIZE]]
            train_d2 = d2_X[c2[:BATCH_SIZE]]

        if method=='bfs':
            G1 = nx.from_scipy_sparse_matrix(d1.obsp['connectivities'])
            G2 = nx.from_scipy_sparse_matrix(d2.obsp['connectivities'])
            n_neighbors = 5
            dMax = min(d1.shape[0], d2.shape[0])
            rIdx = np.random.randint(0, dMax, size=1)
    
            D = 5
            c1 = [int(rIdx)] + [v for u, v in nx.bfs_edges(G1, int(rIdx), depth_limit=D)]
            c2 = [int(rIdx)] + [v for u, v in nx.bfs_edges(G2, int(rIdx), depth_limit=D)]
        
            BATCH_SIZE = min(len(c1), len(c2))
            if ( BATCH_SIZE < 1024):
                BATCH_SIZE = BATCH_SIZE
            else: BATCH_SIZE = 1024
    
            TRAIN_BUF=60000
            TEST_BUF=10000
            N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
            N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
    
            train_d1 = d1_X[c1[:BATCH_SIZE]]
            train_d2 = d2_X[c2[:BATCH_SIZE]]
        
        if method=='hierarchical':
        
            BATCH_SIZE = 1024
            TRAIN_BUF=60000
            TEST_BUF=10000
            N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
            N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
        
            r1,r2 = preprocessing.hierarchical(d1,d2,embedding)
        
            dMax = min(r1.shape[0], r2.shape[0])
            rIdx = np.random.randint(0, dMax, size=BATCH_SIZE)

            train_d1 = r1[rIdx]
            train_d2 = r2[rIdx]
    
        c_train = np.concatenate((train_d1,train_d2), axis=1)
    
        real_x = np.reshape(c_train, (c_train.shape[0],2,d1_X.shape[1]))
        
        train, test = train_test_split(real_x, test_size=0.3)
    
        # batch datasets
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(train)
         #   .shuffle(TRAIN_BUF)
            .batch(BATCH_SIZE)
        )
        test_dataset = (
            tf.data.Dataset.from_tensor_slices(test)
         #   .shuffle(TEST_BUF)
            .batch(BATCH_SIZE)
        )
    
        return train_dataset, test_dataset , N_TRAIN_BATCHES, N_TEST_BATCHES
    

