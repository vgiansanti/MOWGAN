"""
github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ was used as a reference for the WGAN-GP implementation

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import anndata
from sklearn.model_selection import train_test_split
import sys
from sklearn import linear_model
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import scipy
import os

class train:
    def __init__(self, data, query, n_dim, fill, n_epochs, n_samples, save_name, path):
        self.data = data   # list of anndata
        self.query = query   # list of embeddings
        self.n_dim = n_dim   # number of feature in the embedding --> 15
        self.fill = fill # list of filters
        self.n_epochs = n_epochs # number of training epochs
        self.n_samples = n_samples # number of samples in the generated data
        self.save_name = [] # list of names for MOWGAN data
        self.path = "" # path to the working directory, e.g. 'my_working_directory/'
        
    def train(data, query, save_name, path="", n_dim=15, fill=[512,128], n_epochs=100000, n_samples=5000):    
                
        scaler = {}
        data_tr = {}
        r = {}
        A = {}
        L = {}
        l = {}
        e = {}
        d = {}        
        
        for i in range(len(data)):
            if scipy.sparse.issparse(data[i].X):
                data[i].X = data[i].X.todense()
            data[i].X = np.array(data[i].X).astype('float32')
            data[i].obsm[query[i]] = data[i].obsm[query[i]].astype('float32')
            
            scaler['scaler_%s'%i] = MinMaxScaler().fit(data[i].obsm[query[i]][:,:n_dim])
            data_tr['data_tr%s'%i] = scaler['scaler_%s'%i].transform(data[i].obsm[query[i]][:,:n_dim])
            data[i].obs['spectral_emb'] = SpectralEmbedding(n_components=1,
                                     affinity='precomputed').fit_transform(data[i].obsp['connectivities'])
            r['r%s'%i] = data_tr['data_tr%s'%i][np.argsort(data[i].obs['spectral_emb'])]            
            A['A%s'%i] = sc._utils.get_igraph_from_adjacency(data[i].obsp['connectivities'][np.argsort(data[i].obs['spectral_emb'])].todense(), directed=False)
            L['L%s'%i] = np.array(A['A%s'%i].laplacian(normalized=True))   
            l['l%s'%i],e['e%s'%i] = np.linalg.eig(L['L%s'%i])
            d['d%s'%i] = np.real(e['e%s'%i][:,1])
    
        ####################
        # Definition WGAN #
        ####################

        TRAIN_BUF=60000
        BATCH_SIZE= 256
        TEST_BUF=10000
        N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
        N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE) 

        def train_test(r):
            
            shape_0 = []
            for i in range(len(r)):
                shape_0.append(r['r%s'%i].shape[0])
    
            dMax = min(shape_0)
            
            rIdx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
            
            train_anchor = r['r0'][rIdx]
            reg = linear_model.BayesianRidge()
            reg.fit(train_anchor, d['d0'][rIdx])
            
            train_mod = {}            
            train_mod['train_mod0'] = train_anchor
            for i in range(1,len(r)):
                score = []
                Idx = []
                for j in range(50):
                    idx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
                    s = reg.score(r['r%s'%i][idx],d['d%s'%i][idx])
                    Idx.append(idx)
                    score.append(s)
                
                aIdx = Idx[np.argmax(np.asarray(score))]
                train_mod['train_mod%s'%i] = r['r%s'%i][aIdx]
            
            c_train = np.concatenate(list(train_mod.values()),axis=1)            
            real_x = np.reshape(c_train, (c_train.shape[0],len(data),n_dim))
            
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
                z_samp = tf.random.normal([x.shape[0],len(data), self.n_Z])
        
                    # run noise through generator
                x_gen = self.generate(z_samp)

                    # discriminate x and x_gen
                logits_x = self.discriminate(x)
                logits_x_gen = self.discriminate(x_gen)

                    # gradient penalty
                d_regularizer = self.gradient_penalty(x, x_gen)
        
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
    
                tf.keras.layers.Conv1D(filters=512, kernel_size=len(data), strides=1, padding='same', activation="relu"),    
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(filters=128, kernel_size=len(data), strides=1, padding='same', activation="relu"),    
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(filters=n_dim, kernel_size=len(data), strides=1, padding='same', activation="relu"),
            ]

        discriminator = [
                tf.keras.layers.InputLayer(input_shape=(len(data),n_dim)),
                tf.keras.layers.Conv1D(filters=128, kernel_size=len(data), strides=1, padding='same', activation="relu"),
                tf.keras.layers.Conv1D(filters=512, kernel_size=len(data), strides=1, padding='same', activation="relu"),
                tf.keras.layers.Dense(units=1),
            ]
    
        # optimizers
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9, epsilon=1e-07, amsgrad=True)
        disc_optimizer = tf.keras.optimizers.RMSprop(0.0005)

        # a pandas dataframe to save the loss information to
        losses = pd.DataFrame(columns = ['disc_loss1', 'gen_loss'])
    
        model = get_model()
        
        n_epochs = n_epochs

        losses_disc = []
        losses_gen = []
    
        for epoch in range(n_epochs):

            train_dataset, test_dataset, N_TRAIN_BATCHES, N_TEST_BATCHES = train_test(r)   
    
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
                                          
        df = pd.DataFrame(losses_disc)
        df.to_csv(path+'critic_loss.csv', index=False,header=False)
        df = pd.DataFrame(losses_gen)
        df.to_csv(path+'gen_loss.csv', index=False,header=False)

        model.save_weights(path+'MOWGAN_model', save_format='tf')        
        
        samples = model.generate(tf.random.normal(shape=(n_samples,len(data), N_Z)))
        
        for i in range(len(data)):
            neigh = KNeighborsRegressor(n_neighbors=2)
            neigh.fit(data[i].obsm[query[i]][:,:n_dim], data[i].X)
            
            data_adata = np.array(samples[:n_samples, i])            
            data_obsm = scaler['scaler_%s'%i].inverse_transform(data_adata)            
            data_layers = neigh.predict(data_obsm)
            
            anndata_MOWGAN = anndata.AnnData(data_layers)
            anndata_MOWGAN.obsm[query[i]] = data_obsm
            anndata_MOWGAN.var_names = data[i].var_names
            
            if save_name == []:
                save_name = ['anndata_%i'%(i+1) for i in range(len(data))]
 
            anndata_MOWGAN.write(path+save_name[i]+'.h5ad')

class train_batch:
    def __init__(self, data, query, batch, n_dim, fill, n_epochs, n_samples, save_name, path):
        self.data = data   # list of anndata
        self.query = query   # list of embeddings
        self.batch = [] # list of batch names to use in training
        self.n_dim = n_dim   # number of feature in the embedding --> 15
        self.fill = fill # list of filters
        self.n_epochs = n_epochs # number of training epochs
        self.n_samples = n_samples # number of samples in the generated data
        self.save_name = [] # list of names for MOWGAN data
        self.path = "" # path to the working directory, e.g. 'my_working_directory/'
        
    def train(data, query, batch, save_name, path="", n_dim=15, fill=[512,128], n_epochs=100000, n_samples=5000):    
        for i in range(len(data)):            
            
            rename_batch = {}
            
            if len(batch)==1:
                for b in range(len(data[i].obs[batch[0]].cat.categories)):
                    rename_batch[data[i].obs[batch[0]].cat.categories[b]] = str(b)
                    data[i].obs['batch_train'] = (
                    data[i].obs[batch[0]]
                    .map(rename_batch)
                    .astype('category')
                    )
            else:
                for b in range(len(data[i].obs[batch[i]].cat.categories)):
                    rename_batch[data[i].obs[batch[i]].cat.categories[b]] = str(b)
            
                data[i].obs['batch_train'] = (
                data[i].obs[batch[i]]
                .map(rename_batch)
                .astype('category')
                )
                 
            if scipy.sparse.issparse(data[i].X):
                data[i].X = data[i].X.todense()    
            
            data[i].X = np.array(data[i].X).astype('float32')
            data[i].obsm[query[i]] = data[i].obsm[query[i]].astype('float32')
        
        for j in range(len(data[0].obs['batch_train'].value_counts())):
            print(j)
            data_batch = {}
            scaler = {}
            data_tr = {}
            r = {}
            A = {}
            L = {}
            l = {}
            e = {}
            d = {} 
            for i in range(len(data)):
                
                data_batch['data{0}_{1}'.format(i,j)] = data[i][data[i].obs['batch_train']==str(j)]
                scaler['scaler{0}_{1}'.format(i,j)] = MinMaxScaler().fit(data_batch['data{0}_{1}'.format(i,j)].obsm[query[i]][:,:n_dim])
                data_tr['data_tr{0}_{1}'.format(i,j)] = scaler['scaler{0}_{1}'.format(i,j)].transform(data_batch['data{0}_{1}'.format(i,j)].obsm[query[i]][:,:n_dim])
                data_batch['data{0}_{1}'.format(i,j)].obs['spectral_emb'] = SpectralEmbedding(n_components=1,
                                     affinity='precomputed').fit_transform(data_batch['data{0}_{1}'.format(i,j)].obsp['connectivities'])
                r['r{0}_{1}'.format(i,j)] = data_tr['data_tr{0}_{1}'.format(i,j)][np.argsort(data_batch['data{0}_{1}'.format(i,j)].obs['spectral_emb'])]            
                A['A{0}_{1}'.format(i,j)] = sc._utils.get_igraph_from_adjacency(data_batch['data{0}_{1}'.format(i,j)].obsp['connectivities'][np.argsort(data_batch['data{0}_{1}'.format(i,j)].obs['spectral_emb'])].todense(), directed=False)
                L['L{0}_{1}'.format(i,j)] = np.array(A['A{0}_{1}'.format(i,j)].laplacian(normalized=True))   
                l['l{0}_{1}'.format(i,j)],e['e{0}_{1}'.format(i,j)] = np.linalg.eig(L['L{0}_{1}'.format(i,j)])
                d['d{0}_{1}'.format(i,j)] = np.real(e['e{0}_{1}'.format(i,j)][:,1])
                
            ####################
            # Definition WGAN #
            ####################

            TRAIN_BUF=60000
            BATCH_SIZE= 256
            TEST_BUF=10000
            N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
            N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE) 

            def train_test(r):
                
                shape_0 = []
                for i in range(len(r)):
                    shape_0.append(r['r{0}_{1}'.format(i,j)].shape[0])
    
                dMax = min(shape_0)
            
                rIdx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
            
                train_anchor = r['r0_{0}'.format(j)][rIdx]
                reg = linear_model.BayesianRidge()
                reg.fit(train_anchor, d['d0_{0}'.format(j)][rIdx])
            
                train_mod = {}            
                train_mod['train_mod0_{0}'.format(j)] = train_anchor
                for i in range(1,len(r)):
                    score = []
                    Idx = []
                    for k in range(50):
                        idx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
                        s = reg.score(r['r{0}_{1}'.format(i,j)][idx],d['d{0}_{1}'.format(i,j)][idx])
                        Idx.append(idx)
                        score.append(s)
                
                    aIdx = Idx[np.argmax(np.asarray(score))]
                    train_mod['train_mod{0}_{1}'.format(i,j)] = r['r{0}_{1}'.format(i,j)][aIdx]
            
                c_train = np.concatenate(list(train_mod.values()),axis=1)            
                real_x = np.reshape(c_train, (c_train.shape[0],len(data),n_dim))
            
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
                    z_samp = tf.random.normal([x.shape[0],len(data), self.n_Z])
        
                    # run noise through generator
                    x_gen = self.generate(z_samp)

                    # discriminate x and x_gen
                    logits_x = self.discriminate(x)
                    logits_x_gen = self.discriminate(x_gen)

                    # gradient penalty
                    d_regularizer = self.gradient_penalty(x, x_gen)
        
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
    
                tf.keras.layers.Conv1D(filters=512, kernel_size=len(data), strides=1, padding='same', activation="relu"),    
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(filters=128, kernel_size=len(data), strides=1, padding='same', activation="relu"),    
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv1D(filters=n_dim, kernel_size=len(data), strides=1, padding='same', activation="relu"),
            ]

            discriminator = [
                tf.keras.layers.InputLayer(input_shape=(len(data),n_dim)),
                tf.keras.layers.Conv1D(filters=128, kernel_size=len(data), strides=1, padding='same', activation="relu"),
                tf.keras.layers.Conv1D(filters=512, kernel_size=len(data), strides=1, padding='same', activation="relu"),
                tf.keras.layers.Dense(units=1),
            ]
    
            # optimizers
            gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9, epsilon=1e-07, amsgrad=True)
            disc_optimizer = tf.keras.optimizers.RMSprop(0.0005)# train the model

            # a pandas dataframe to save the loss information to
            losses = pd.DataFrame(columns = ['disc_loss1', 'gen_loss'])
    
            model = get_model()
        
            n_epochs = n_epochs

            losses_disc = []
            losses_gen = []
    
            for epoch in range(n_epochs):
    
                train_dataset, test_dataset, N_TRAIN_BATCHES, N_TEST_BATCHES = train_test(r)   
    
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
                                      
            df = pd.DataFrame(losses_disc)
            df.to_csv(path+'critic_loss_batch_{0}'.format(j)+'.csv', index=False,header=False)
            df = pd.DataFrame(losses_gen)
            df.to_csv(path+'gen_loss_batch_{0}'.format(j)+'.csv', index=False,header=False)

            model.save_weights(path+'MOWGAN_model_batch_{0}'.format(j), save_format='tf')

            samples = model.generate(tf.random.normal(shape=(n_samples,len(data), N_Z)))
            
            for i in range(len(data)):
                neigh = KNeighborsRegressor(n_neighbors=2)
                neigh.fit(data[i].obsm[query[i]][:,:n_dim], data[i].X)
            
                data_adata = np.array(samples[:n_samples, i])            
                data_obsm = scaler['scaler{0}_{1}'.format(i,j)].inverse_transform(data_adata)            
                data_layers = neigh.predict(data_obsm)
            
                anndata_MOWGAN = anndata.AnnData(data_layers)
                anndata_MOWGAN.obsm[query[i]] = data_obsm
                anndata_MOWGAN.var_names = data[i].var_names
            
                anndata_MOWGAN.write(path+'anndata{0}_{1}'.format(i,j)+'.h5ad')  
                
        for i in range(len(data)):
            if path=="":
                prefixed = [filename for filename in os.listdir('.') if filename.startswith("anndata"+str(i))]
            else:
                prefixed = [filename for filename in os.listdir(path) if filename.startswith("anndata"+str(i))]
            data_read = []
            for j in range(len(prefixed)):
                data_read.append(sc.read(path+prefixed[j]))
                data_conc = data_read[0].concatenate(data_read[1:])
                if save_name == []:
                    data_conc.write(path+'data'+str(i)+'_MOWGAN.h5ad')
                else:
                    data_conc.write(path+save_name[i]+'_MOWGAN.h5ad')

