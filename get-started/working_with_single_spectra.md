---
title: Working with single spectrum
layout: default
parent: Get Started
nav_order: 2
---

## __Working with single spectrumS__
Preprocessing techniques in scikit-learn are primarily designed to work with 2D arrays, where each row represents a sample and each column represents a feature (i.e., matrices). However, in spectroscopy, single spectra are often of interest, which are represented as 1D arrays (i.e., vectors). To apply scikit-learn and chemotools techniques to single spectra, they need to be reshaped into 2D arrays (i.e., a matrix with one row). To achieve this, you can use the following code that reshapes a 1D array into a 2D array with a single row:

```python
import numpy as np
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection()
spectra_msc = msc.fit_transform(spectra.reshape(1, -1))
```
The ```.reshape(1, -1)``` method is applied to the 1D array ```spectra```, which is converted into a 2D array with a single row.

