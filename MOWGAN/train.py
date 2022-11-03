"""
github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ was used as a reference for the WGAN-GP implementation

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
import pandas as pd
import scanpy as sc
import anndata
from sklearn.model_selection import train_test_split
import scipy
import sys
import scipy.cluster
import scipy.spatial
from sklearn import linear_model
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor


# the anndata must already have the embedding slot

class train:
    def __init__(self, adata1, adata2, adata1_query, adata2_query, n_dim, n_epochs):
        self.adata1 = adata1   # anndata modality 1
        self.adata2 = adata2   # anndata modality 2
        self.adata1_query = adata1_query   # embedding modality 1 --> 'X_pca'
        self.adata2_query = adata2_query   # embedding modality 2 --> 'X_pca'
        self.n_dim = n_dim   # number of feature in the embedding --> 15
        self.n_epochs = n_epochs
        
    def train(adata1, adata2, adata1_query, adata2_query, n_dim=15, n_epochs=100000):
        
        # Embedding preparation --> select the embedding, number of feature, scale
        
        query1 = np.float32(adata1.obsm[adata1_query][:,:int(n_dim)])
        query2 = np.float32(adata2.obsm[adata2_query][:,:int(n_dim)])
        
        scaler_adata1 = MinMaxScaler()
        scaler_adata2 = MinMaxScaler()

        scaler_adata1.fit(query1)
        scaler_adata2.fit(query2)                

        adata1_tr = scaler_adata1.transform(query1)
        adata2_tr = scaler_adata2.transform(query2)
        
        # Sort embedding

        embedding_adata1 = SpectralEmbedding(n_components=1,affinity='precomputed').fit_transform(adata1.obsp['connectivities'])
        embedding_adata2 = SpectralEmbedding(n_components=1,affinity='precomputed').fit_transform(adata2.obsp['connectivities'])

        adata1.obs['embedding'] = embedding_adata1
        adata2.obs['embedding'] = embedding_adata2
    
        r1 = adata1_tr[np.argsort(adata1.obs['embedding'])]
        r2 = adata2_tr[np.argsort(adata2.obs['embedding'])]
    
        R = sc._utils.get_igraph_from_adjacency(adata1.obsp['connectivities'][np.argsort(adata1.obs['embedding'])], directed=False)
        A = sc._utils.get_igraph_from_adjacency(adata2.obsp['connectivities'][np.argsort(adata2.obs['embedding'])], directed=False)

        Lr = R.laplacian(normalized=True)
        Lr = np.array(Lr)
        lr,er = np.linalg.eig(Lr)

        La = A.laplacian(normalized=True)
        La = np.array(La)
        la,ea = np.linalg.eig(La)

        d1_sort = er[:,1]
        d2_sort = ea[:,1]

                
        TRAIN_BUF=60000
        BATCH_SIZE= 128
        TEST_BUF=10000    
        N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
        N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)  
        
        def train_test(d1,d2): # sono r1 e r2
            dMax = min(d1.shape[0], d2.shape[0])
            rIdx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
    
            train_d1 = d1[rIdx]
    
            reg = linear_model.BayesianRidge()    
            reg.fit(train_d1, d1_sort[rIdx])
    
            score = []
            Idx = []

            for i in range(50):
                idx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
                s = reg.score(d2[idx],d2_sort[idx])
                Idx.append(idx)
                score.append(s)
    
            aIdx = Idx[np.argmax(np.asarray(score))]
            train_d2 = d2[aIdx]
        
            c_train = np.concatenate((train_d1,train_d2), axis=1)
    
            real_x = np.reshape(c_train, (c_train.shape[0],2,d1.shape[1]))
        
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
        
        
        class WGAN(tf.keras.Model):

            def __init__(self, **kwargs):
                super(WGAN, self).__init__()
                self.__dict__.update(kwargs)

                self.gen = tf.keras.Sequential(self.gen)
                self.disc = tf.keras.Sequential(self.disc)
        
            def generate(self, z):
                return self.gen(z)

            def discriminate(self, x):
                return self.disc(x)

            def compute_loss(self, x):
                """ passes through the network and computes loss
                """
                ### pass through network
                # generating noise from a uniform distribution
                z_samp = tf.random.normal([x.shape[0],2, self.n_Z])
        
                # run noise through generator
                x_gen = self.generate(z_samp)

                # discriminate x and x_gen
                logits_x = self.discriminate(x)
                logits_x_gen = self.discriminate(x_gen)

                # gradient penalty
                d_regularizer = self.gradient_penalty(x, x_gen)
        
                ### aggiungere la wassertein loss tra x_rna_gen e x_atac_gen
                ### losses
                disc_loss1 = (
                    tf.reduce_mean(logits_x)
                    - tf.reduce_mean(logits_x_gen)
                    + d_regularizer * self.gradient_penalty_weight
                )

            # losses of fake with label "1"
                gen_loss = tf.reduce_mean(logits_x_gen)  
                   
                return disc_loss1, gen_loss

            def compute_gradients(self, x):
                """ passes through the network and computes loss
                """
            ### pass through network
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape1: 
                    disc_loss1, gen_loss = self.compute_loss(x)

            # compute gradients
                gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

                disc_gradients1 = disc_tape1.gradient(disc_loss1, self.disc.trainable_variables)

                return gen_gradients, disc_gradients1 

            def apply_gradients(self, gen_gradients, disc_gradients1): 

                self.gen_optimizer.apply_gradients(
                    zip(gen_gradients, self.gen.trainable_variables)
            )
                self.disc_optimizer.apply_gradients(
                    zip(disc_gradients1, self.disc.trainable_variables)
            )

            def gradient_penalty(self, x, x_gen):
        
                epsilon = tf.random.uniform([x.shape[0],1, 1], 0.0, 1.0)
                x_hat = epsilon * x + (1 - epsilon) * x_gen
        
                with tf.GradientTape() as t:
                    t.watch(x_hat)
                    d_hat = self.discriminate(x_hat)
                gradients = t.gradient(d_hat, x_hat)
                ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
                d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
                return d_regularizer


            @tf.function
            def train(self, train_x):
                gen_gradients, disc_gradients1 = self.compute_gradients(train_x)
                self.apply_gradients(gen_gradients, disc_gradients1)
        
        def get_model():
              return WGAN(gen = generator,
            disc = discriminator,
            gen_optimizer = gen_optimizer,
            disc_optimizer = disc_optimizer,
            n_Z = N_Z,
            gradient_penalty_weight = 10.0,
                      name='WGAN')
              
        N_Z = 1024

        generator = [
    
            tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same', activation="relu"),    
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same', activation="relu"),    
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=n_dim, kernel_size=2, strides=1, padding='same', activation="relu"),
        ]

        discriminator = [
            tf.keras.layers.InputLayer(input_shape=(2,n_dim)),
            tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same', activation="relu"),
            tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same', activation="relu"),
            tf.keras.layers.Dense(units=1),
        ]
    
    # optimizers
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9, epsilon=1e-07)#, amsgrad=True)
        disc_optimizer = tf.keras.optimizers.RMSprop(0.0005)# train the model

    # a pandas dataframe to save the loss information to
        losses = pd.DataFrame(columns = ['disc_loss1', 'gen_loss'])
    
        model = get_model()
        
        n_epochs = n_epochs

        losses_disc = []
        losses_gen = []
    
        for epoch in range(n_epochs):

            train_dataset, test_dataset, N_TRAIN_BATCHES, N_TEST_BATCHES = train_test(r1,r2)   
    
        # train
            for batch, train_x in tqdm(
                zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
            ):
                model.train(train_x)

        # test on holdout
            loss = []
            for batch, test_x in tqdm(
                zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
            ):
                loss.append(model.compute_loss(train_x))
                losses.loc[len(losses)] = np.mean(loss, axis=0)
    
            losses_disc.append(losses.disc_loss1.values[-1])
            losses_gen.append(losses.gen_loss.values[-1])
    
        # plot results
            print(
                "Epoch: {} | disc_loss1: {} | gen_loss: {} ".format(
                    epoch, losses.disc_loss1.values[-1], losses.gen_loss.values[-1]
                )
            )
    
        #    for N in range(10000,n_epochs+1, 10000):
        #        if epoch == N:
        #            model.save_weights('MOWGAN_IPS_BATCH_double_'+str(i)+'_'+name+'_'+str(N), save_format='tf')
                                      
        df = pd.DataFrame(losses_disc)
        df.to_csv('disc_loss.csv', index=False,header=False)
        df = pd.DataFrame(losses_gen)
        df.to_csv('gen_loss.csv', index=False,header=False)

        model.save_weights('MOWGAN_model', save_format='tf') 

        cells_max = max(adata1.shape[0],adata2.shape[0])
        cells_d1 = min(cells_max,adata1.shape[0])
        cells_d2 = min(cells_max,adata2.shape[0])

        samples = model.generate(tf.random.normal(shape=(cells_max,2, N_Z)))

        data_adata1 = np.array(samples[:cells_d1, 0])
        data_adata2 = np.array(samples[:cells_d2, 1])

        # adata1
        adata1.obsm['X_MOWGAN'] = scaler_adata1.inverse_transform(data_adata1)

        neigh = KNeighborsRegressor(n_neighbors=2)
        neigh.fit(adata1.obsm[adata1_query][:,:int(n_dim)], adata1.X.todense())
        adata1.layers['X_MOWGAN'] = neigh.predict(adata1.obsm['X_MOWGAN'])

        # adata2
        adata2.obsm['X_MOWGAN'] = scaler_adata2.inverse_transform(data_adata2)

        neigh = KNeighborsRegressor(n_neighbors=2) #int(np.sqrt(rna.shape[0]/2))
        neigh.fit(adata2.obsm[adata2_query][:,:int(n_dim)], adata2.X.todense())
        adata2.layers['X_MOWGAN'] = neigh.predict(adata2.obsm['X_MOWGAN'])
        
        adata1_MOWGAN = anndata.AnnData(adata1.layers['X_MOWGAN'])
        adata1_MOWGAN.obsm[adata1_query] = adata1.obsm['X_MOWGAN']
        adata1_MOWGAN.var_names = adata1.var_names
        
        adata2_MOWGAN = anndata.AnnData(adata2.layers['X_MOWGAN'])
        adata2_MOWGAN.obsm[adata2_query] = adata2.obsm['X_MOWGAN']
        adata2_MOWGAN.var_names = adata2.var_names

        adata1_MOWGAN.write('adata1_MOWGAN.h5ad')
        adata2_MOWGAN.write('adata2_MOWGAN.h5ad')
