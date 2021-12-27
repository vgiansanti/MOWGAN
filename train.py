############################
# FUNZIONE PER IL TRAINING #
############################

# Opzioni disponibili: 1) method='random_walk'
#                      2) method='bfs'
#                      3) method='hierarchical'

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd

from MOWGAN.preprocessing import preprocessing
from MOWGAN.model import get_model


### train function
def train(adata1, adata2, n_epochs=10000,method='random_walk',embedding='umap', **kwds):

    losses_disc = []
    losses_gen = []
    losses = pd.DataFrame(columns = ['disc_loss1', 'gen_loss'])

    model = get_model()
    
    adata1_tr,adata2_tr = preprocessing.scaler(adata1,adata2,embedding)
    
    for epoch in range(n_epochs):

        train_dataset, test_dataset, N_TRAIN_BATCHES, N_TEST_BATCHES = preprocessing.train_test(adata1, adata2, 
                                                             adata1_tr, adata2_tr, embedding, method=method)
    
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

        print(
            "Epoch: {} | disc_loss1: {} | gen_loss: {} ".format(
                epoch, losses.disc_loss1.values[-1], losses.gen_loss.values[-1] 
            )
    )
        
    df = pd.DataFrame(losses_disc)
    df.to_csv('disc_loss.csv', index=False,header=False)

    df = pd.DataFrame(losses_gen)
    df.to_csv('gen_loss.csv', index=False,header=False)
    
    model.save_weights('MOWGAN', save_format='tf')