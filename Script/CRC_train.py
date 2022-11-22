# Import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
import pandas as pd
import scanpy as sc
import anndata
from sklearn.model_selection import train_test_split
import networkx as nx
import scipy
import sys
import scipy.cluster
import scipy.spatial
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import SpectralEmbedding
import igraph as ig

rna_read = 'CRC_RNA.h5ad'
atac_read = 'CRC_GET.h5ad'
pc = int(20) # n_pc
fil1 = int(512) # filter1
fil2 = int(128) # filters2

path=path
rna = sc.read(path+rna_read)
atac = sc.read(path+atac_read)

old_to_new = {
'CRC_6':'0',
'CRC_17':'1',
'CRC_39':'2'
}
rna.obs['batch_train'] = (
rna.obs['sample']
.map(old_to_new)
.astype('category')
)

old_to_new = {
'CRC_6':'0',
'CRC_17':'1',
'CRC_39':'2'
}
atac.obs['batch_train'] = (
atac.obs['batch']
.map(old_to_new)
.astype('category')
)

for i in range(len(rna.obs['batch_train'].value_counts())):
    rna_sub = rna[rna.obs['batch_train']==str(i)]
    atac_sub = atac[atac.obs['batch_train']==str(i)]
    
    min_cells = min(rna_sub.shape[0],atac_sub.shape[0])
    sc.pp.subsample(rna_sub, n_obs=min_cells)
    sc.pp.subsample(atac_sub, n_obs=min_cells)
                
    rna_tr = rna_sub.obsm['X_pca'][:,:20]
    atac_tr = np.float32(atac_sub.obsm['X_ttd'][:,:20])
    
    scaler_rna = MinMaxScaler()
    scaler_atac = MinMaxScaler()

    scaler_rna.fit(rna_tr)
    scaler_atac.fit(atac_tr)

    rna_tr = scaler_rna.transform(rna_tr)
    atac_tr = scaler_atac.transform(atac_tr)
    
    embedding_rna = SpectralEmbedding(n_components=1,affinity='precomputed').fit_transform(rna_sub.obsp['connectivities'])
    embedding_atac = SpectralEmbedding(n_components=1,affinity='precomputed').fit_transform(atac_sub.obsp['connectivities'])

    rna_sub.obs['embedding'] = embedding_rna
    atac_sub.obs['embedding'] = embedding_atac
    
    r1 = rna_tr[np.argsort(rna_sub.obs['embedding'])]
    r2 = atac_tr[np.argsort(atac_sub.obs['embedding'])]
    
    R = sc._utils.get_igraph_from_adjacency(rna_sub.obsp['connectivities'][np.argsort(rna_sub.obs['embedding'])], directed=False)
    A = sc._utils.get_igraph_from_adjacency(atac_sub.obsp['connectivities'][np.argsort(atac_sub.obs['embedding'])], directed=False)

    Lr = R.laplacian(normalized=True)
    Lr = np.array(Lr)
    lr,er = np.linalg.eig(Lr)

    La = A.laplacian(normalized=True)
    La = np.array(La)
    la,ea = np.linalg.eig(La)

    d1_leiden = er[:,1]
    d2_leiden = ea[:,1]

    
    TRAIN_BUF=60000
    BATCH_SIZE= 128 #1024 #512 #1024  # 512
    TEST_BUF=10000    
    N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
    N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)    
        
    def train_test(d1,d2): # sono r1 e r2
        dMax = min(d1.shape[0], d2.shape[0])
        rIdx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
    
        train_d1 = d1[rIdx]
        
        reg = linear_model.BayesianRidge()
        reg.fit(train_d1, d1_leiden[rIdx])                   
    
        score = []
        Idx = []
        
        for i in range(50):
            idx = np.sort(np.random.randint(0, dMax, size=BATCH_SIZE))
            s = reg.score(d2[idx],d2_leiden[idx])
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
        """[summary]
        I used github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ as a reference on this.
    
        Extends:
            tf.keras.Model
        """

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

        tf.keras.layers.Conv1D(filters=fil1, kernel_size=2, strides=1, padding='same', activation="relu"),    
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=fil2, kernel_size=2, strides=1, padding='same', activation="relu"),    
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=pc, kernel_size=2, strides=1, padding='same', activation="relu"),
    ]

    discriminator = [
        tf.keras.layers.InputLayer(input_shape=(2,pc)),
        tf.keras.layers.Conv1D(filters=fil2, kernel_size=2, strides=1, padding='same', activation="relu"),
        tf.keras.layers.Conv1D(filters=fil1, kernel_size=2, strides=1, padding='same', activation="relu"),
        tf.keras.layers.Dense(units=1),
    ]
    
# optimizers
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.9, epsilon=1e-07, amsgrad=True)
    disc_optimizer = tf.keras.optimizers.RMSprop(0.0005)# train the model

# a pandas dataframe to save the loss information to
    losses = pd.DataFrame(columns = ['disc_loss1', 'gen_loss'])

    model = get_model()

    n_epochs = 50000

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
    
        for N in range(10000,n_epochs+1, 10000):
            if epoch == N:
                model.save_weights(path+'MOWGAN_CRC_SPECTRAL_BATCH_'+str(i)+'_'+str(N), save_format='tf')
                
    
    path1 = '/beegfs/scratch/ric.cosr/giansanti.valentina/MOWGAN_CRC/'
    name = str(pc)+'_'+str(fil1)+'_'+str(fil2)

    df = pd.DataFrame(losses_disc)
    df.to_csv(path1+'disc_loss_SPECTRAL_BATCH_'+str(i)+'_CRC.csv', index=False,header=False)
    df = pd.DataFrame(losses_gen)
    df.to_csv(path1+'gen_loss_SPECTRAL_BATCH_'+str(i)+'_CRC.csv', index=False,header=False)

    model.save_weights(path1+'MOWGAN_CRC_SPECTRAL_BATCH_'+str(i), save_format='tf')         
