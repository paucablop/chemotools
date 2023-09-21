---
title: Datasets
layout: default
parent: Get Started
nav_order: 3
---

# __Datasets__
The ```chemotools``` package includes a number of datasets that can be used to test the package and to learn how to use it. The datasets are stored in the ```chemotools.datasets``` module and can be accessed using loading functions. The available datasets are:

- The Fermentation Dataset, which contains spectra collected during a fermentation process.

- The Coffee Dataset, which contains spectra collected from different coffee samples.


## The Fermentation Dataset
The Fermentation Dataset contains spectra collected during a fermentation process. The spectra are collected using attenuated total reflectance Fourier transform infrared spectroscopy (ATR-FTIR). The Fermentation Dataset consists of two sets of spectra:

##### __THE TRAIN SET__ 
The train set consists of 21 synthetic spectra and their reference glucose concentrations measured by high-performance liquid chromatography (HPLC).

The train set can be loaded using the ```load_fermentation_train``` function:

```python
from chemotools.datasets import load_fermentation_train

X_train, y_train = load_fermentation_train()
```

{: .note}
> Learn how to train a PLS model using the Fermentation Dataset [on the training Guide](https://paucablop.github.io/chemotools/get-started/brewing_regressor.html).

##### __THE TEST SET__
The test set consists of over 1000 spectra collected in _real-time_ during a fermentation process. The spectra are collected every 1.25 minutes for a period of several hours. Moreover, 35 reference glucose concentrations measured by HPLC are available for the test set. These reference concentrations where measured every hour during the fermentation and can be used to evaluate the performance of a model.

The test set can be loaded using the ```load_fermentation_test``` function:

```python
from chemotools.datasets import load_fermentation_test

X_test, y_test = load_fermentation_test()
```

## The Coffee Dataset
The Coffee Dataset contains spectra collected from different coffee samples from different countries. The spectra are collected using attenuated total reflectance Fourier transform infrared spectroscopy (ATR-FTIR). The Coffee Dataset can be loaded using the ```load_coffee``` function:

```python
from chemotools.datasets import load_coffee

spectra, labels = load_coffee()
```

{: .note}
> Learn how to train a PLS-DA classification model using the Coffee Dataset [on the training Guide](https://paucablop.github.io/chemotools/get-started/coffee_spectra_classifier.html).