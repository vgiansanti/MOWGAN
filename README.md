# MOWGAN: Multi-Omics Wasserstein Generative Adversarial Network

![Alt text](images/Mowgan_round.tiff)

MOWGAN is a deep learning framework for the generation of synthetic paired multiomics single-cell datasets. The core component is a single Wasserstein Generative Adversarial Network with gradient penalty (WGAN-GP). Inputs are data from multi-omics experiment in unpaired observation. MOWGAN infers the most reliable coupling and train the network to learn the internal structur of the modalities.

Once trained, the generative network is used to produce a new dataset where the observations are matched between all modalities. The synthetic dataset can be used for downstream analysis, first of all to bridge the original unpaired data.

## Tutorials

The notebook "MOWGAN_training.ipynb" shows, on toy datasets, how to apply MOWGAN to learn and generated coupled datasets.

## Cheatsheet

MOWGAN main function is the train(). How to use it:

```
from MOWGAN.train import train

train.train(data, query, n_dim, fill, n_epochs, n_samples, save_name=[])
```

where:
* data -> list of anndata objects (e.g., data=[data1, data2])
* query -> list of embeddings (e.g., query=['X_pca','X_umap'])
* n_dim -> number of feature to consider in the embeddings (by default, n_dim=15)
* fill -> list of filters for the neural network layers (by default, fill=[512,128])
* n_epochs -> number of training epochs (by default, n_epochs=100000)
* n_samples -> number of samples in the generated data (by default, n_samples=5000)
* save_name -> list of names for MOWGAN data (default is save_name=[], data are saved as "anndata_1.h5ad", "anndata_2.h5ad", etc.)

## MOWGAN workflow

### Step 1: Train the model
![Alt text](images/Figure1.png)



### Step 2: Train the model

## Contact
MOWGAN is maintained by Valentina Giansanti (giansanti.valentina@hsr.it) and Davide Cittaro (cittaro.davide@hsr.it). Please, reach us for problems, comments or suggestions.
