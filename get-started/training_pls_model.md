---
title: Training a PLS model using scikit-learn
layout: default
parent: Get Started
nav_order: 3
---


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
