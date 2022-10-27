# MOWGAN - Multi-Omics Wasserstein Generative Adversarial Network

![Alt text](images/Mowgan_round.tiff)

MOWGAN is a deep learning framework for the generation of synthetic paired multiomics single-cell datasets. The core component is a single Wasserstein Generative Adversarial Network with gradient penalty (WGAN-GP). Inputs are data from multi-omics experiment in unpaired observation. MOWGAN infers the most reliable coupling and train the network to learn the internal structur of the modalities.

Once trained, the generative network is used to produce a new dataset where the observations are matched between all modalities. The synthetic dataset can be used for downstream analysis, first of all to bridge the original unpaired data.

## Tutorials

## Cheatsheet

Relevant functions:

## MOWGAN workflow

### Step 1: Train the model
![Alt text](images/Figure1.png)



### Step 2: Train the model

## Contact
MOWGAN is maintained by Valentina Giansanti (giansanti.valentina@hsr.it) and Davide Cittaro (cittaro.davide@hsr.it). Please, reach us for problems, comments or suggestions.
