# Anomaly Detection (AD) for Hyperspectral Images (HSI)

This repository includes the programming code with the experiments described in the Master's Thesis defended for the **Master's Degree on Artificial Intelligence**, at the **International University of Valencia** (**VIU**).
- Author: **Telmo Subirá Rodríguez**
- Title (spanish): **Modelos Basados en Transformer para la Detección de Anomalías en Imágenes Hiperespectrales**
- Title (english): **Transformer-based Models for Anomaly Detection on Hyperspectral Images**
  
The code distributed in this repository uses code from the two following sources:
- [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch) by [Phil Wang](https://github.com/lucidrains)
- [Constrained Unsupervised Anomaly Segmentation of Brain Lesions](https://github.com/jusiro/constrained_anomaly_segmentation) by [Julio Silva-Rodríguez](https://github.com/jusiro)

## Summary
The repository contains the code for two different AD strategies:
- Vision Transformer (**ViT**) Masked Autoencoder (**MAE**) model.
- Convolutional Variational Autoencoder (**CVAE**) model.

Both strategies include python scripts (*.py*) and interactive notebooks (*.ipynb*) to create, train and test the models.
Additional scripts and functions allow the user to prepare the training/test data from the free-download available [**ABU dataset**](http://xudongkang.weebly.com/data-sets.html).

## ViT-MAE model
### Instructions
Download the [**ABU dataset**](http://xudongkang.weebly.com/data-sets.html) and store the *.mat* files inside the *ViT-MAE/data* directory.
Run the *model_train_test.ipynb* file to train a model and test it on a HSI.

### Core files
- *vit.py*. Create the ViT model for the MAE, adapted from [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch).
- *mae.py*. Create the MAE model using the ViT as the encoder, adapted from [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch).
- *model_train_test.ipynb*. Interactive notebook to create and train a model, showing evaluation results.

### Auxiliary files
- *HSI_utils.py*. Auxiliar functions to manage the dataset, plot graphs, etc.
- *losses.py*. Auxiliar functions for computing loss functions during the training of the models.

## CVAE model
### Instructions
Download the [**ABU dataset**](http://xudongkang.weebly.com/data-sets.html) and store the *.mat* files inside the *CVAE/data* directory.
First run the *create_models_full_HSI.ipynb* notebook to create a VAE model, train it over one of the HSI from the dataset, and later save evaluation results.
To visualize the results and generate metrics, run the *test_AD_full_HSI.ipynb*.

### Core files
- *conv_models_AD_full_HSI.py*. This file includes the definition of the convolutional **Encoder** and **Decoder** classes to create **VAE** architectures.
- *VAE_model_trainer_full_HSI.py*. This file includes the definition of the VAE class and its training loop as a method.
- *create_models_full_HSI.ipynb*. Interactive notebook for creating the VAE model, launching the training and testing over one HSI. Test results are saved as files.
- *test_AD_full_HSI.ipynb*. Notebook for reading the test results and generate metrics over one HSI.

### Auxiliary files
- *test_enc_dec.ipynb*. Quick creation and test of dummy encoder and decoder models. For debugging.
- *HSI_utils.py*. Auxiliar functions to manage the dataset, plot graphs, etc.
- *losses.py*. Auxiliar functions for computing loss functions during the training of the models.



