---
title: Scikit learn integration
layout: default
parent: Docs
nav_order: 1
---


# __scikit-learn integration__

This section describes how to use the preprocessing techniques in this package with ```scikit-learn```, highlighting some common situations that may arise when working with spectroscopic data and ```scikit-learn```.

- [Working with single spectra](#working-with-single-spectra)
- [Pipelines integration with ```scikit-learn```](#pipelines-integration-with-scikit-learn)
- [Training a PLS model using ```scikit-learn```](#training-a-pls-model)


## __Working with single spectra__
Preprocessing techniques in scikit-learn are primarily designed to work with 2D arrays, where each row represents a sample and each column represents a feature (i.e., matrices). However, in spectroscopy, single spectra are often of interest, which are represented as 1D arrays (i.e., vectors). To apply scikit-learn and chemotools techniques to single spectra, they need to be reshaped into 2D arrays (i.e., a matrix with one row). To achieve this, you can use the following code that reshapes a 1D array into a 2D array with a single row:

```python
import numpy as np
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection()
spectra_msc = msc.fit_transform(spectra.reshape(1, -1))
```
The ```.reshape(1, -1)``` method is applied to the 1D array ```spectra```, which is converted into a 2D array with a single row.


## __Pipelines integration with scikit-learn__
All preprocessing techniques in this package are compatible with ```scikit-learn``` and can be used in pipelines. For example, the following code creates a pipeline that performs:

- Whittaker smoothing
- AirPLS baseline correction
- Multiplicative scatter correction
- Mean centering

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.baseline import AirPls
fom chemotools.scatter import MultiplicativeScatterCorrection
from chemotools.smooth import WhittakerSmooth

pipeline = make_pipeline(
    WhittakerSmooth(),
    AirPls(),
    MultiplicativeScatterCorrection(),
    StandardScaler(with_std=False),
)
```
Now the pileline can be visualized, which will show the sequence of preprocessing techniques that will be applied in the pipeline.

<iframe src="figures/pipeline_visual.html" width="100%" style="border: none;"></iframe>

Once the pipeline is created, it can be used to fit and transform the spectra using the ```.fit_transform()``` method.

```python
spectra_transformed = pipeline.fit_transform(spectra)
```
This will produce the following output:

<iframe src="figures/pipeline.html" width="800px" height="500px" style="border: none;"></iframe>



## __Training a PLS model using scikit-learn__

The following code shows how to train a PLS model using ```scikit-learn```. The following preprocessing are used:

- Linear correction
- Savitzky-Golay derivative
- Range cut by index
- Mean centering

Then a PLS model with 2 components is trained (using the ```.fit()``` method) and used to predict the independent values of the test set (using the ```.predict()``` method).

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.baseline import LinearCorrection
from chemotools.derivative import SavitzkyGolay
from chemotools.variable_selection import RangeCutByIndex


pipeline = make_pipeline(
    LinearCorrection(),
    SavitzkyGolay(),
    RangeCutByIndex(0, 300),
    StandardScaler(with_mean=True, with_std=False),
    PLSRegression(n_components=2)
)

prediction = pipeline.fit(train_spectra, train_reference).predict(test_spectra)
```
