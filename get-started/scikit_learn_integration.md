---
title: Scikit-learn integration
layout: default
parent: Get Started
nav_order: 2
---

# __Scikit-learn integration__

This page shows how to use ```chemotools``` in combination with ```scikit-learn```. The following topics are covered:

- [Working with single spectra](#working-with-single-spectra)
- [Working with pipelines](#working-with-pipelines)
- [Working with pandas DataFrames](#working-with-pandas-dataframes)
- [Persisting your pipelines](#persisting-your-pipelines)

## __Working with single spectra__
Preprocessing techniques in scikit-learn are primarily designed to work with 2D arrays, where each row represents a sample and each column represents a feature (i.e., matrices). However, in spectroscopy, single spectra are often of interest, which are represented as 1D arrays (i.e., vectors). To apply scikit-learn and chemotools techniques to single spectra, they need to be reshaped into 2D arrays (i.e., a matrix with one row). To achieve this, you can use the following code that reshapes a 1D array into a 2D array with a single row:

```python
import numpy as np
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection()
spectra_msc = msc.fit_transform(spectra.reshape(1, -1))
```
The ```.reshape(1, -1)``` method is applied to the 1D array ```spectra```, which is converted into a 2D array with a single row.

## __Working with pipelines__
All preprocessing techniques in this package are compatible with ```scikit-learn``` and can be used in pipelines. For example, the following code creates a pipeline that performs:

- Whittaker smoothing
- AirPLS baseline correction
- Multiplicative scatter correction
- Mean centering

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.baseline import AirPls
from chemotools.scatter import MultiplicativeScatterCorrection
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

## __Working with pandas DataFrames__
For the ```pandas.DataFrame``` lovers. By default, all ```scikit-learn``` and ```chemotools``` transformers output ```numpy.ndarray```. However, now it is possible to configure your ```chemotools``` preprocessing methods to produce ```pandas.DataFrame``` objects as output. This is possible after implementing the new ```set_output()``` API from ```scikit-learn```>= 1.2.2 ([documentation](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html)). The same API implemented in other ```scikit-learn``` preprocessing methods like the ```StandardScaler()``` is now available for the ```chemotools``` transformers. 

{: .warning }
> Right now, the ```set_output()``` API is not available for the ```RangeCut()``` method. This is because the ```RangeCut()``` method changes the names of the columns in the input array, which is not compatible with the ```set_output()``` API from ```scikit-learn```. We will look to fix this in future releases.

Below there are two examples of how to use this new API:

#### __Example 1: Using the ```set_output()``` API with a single preprocessing method__

### 1. Load your spectral data as a ```pandas.DataFrame```.
First load your spectral data. In this case we assume a file called ```spectra.csv``` where each row represents a spectrum and each column represents a wavenumbers.

```python
import pandas as pd
from chemotools.baseline import AirPls

# Load your data as a pandas DataFrame
spectra = pd.read_csv('data/spectra.csv', index_col=0)
```

The ```spectra``` variable is a ```pandas.DataFrame``` object with the indices representing the sample names and the columns representing the wavenumbers. The first 5 rows of the ```spectra``` DataFrame look like this:

<iframe src="figures/spectra.html" width="100%" style="border: none;"></iframe>

### 2. Create a ```chemotools``` preprocessing object and set the output to ```pandas```.
Next, we create the ```AirPls``` object and set the output to ```pandas```.

```python
# Create an AirPLS object and set the output to pandas
airpls = AirPls().set_output(transform='pandas')
```
The ```set_output()``` method accepts the following arguments:

- ```transform```: The output format. Can be ```'pandas'``` or ```'default'``` (the default format will output a ```numpy.ndarray```).


### 3. Fit and transform the spectra

```python
# Fit and transform the spectra
spectra_airpls = airpls.fit_transform(spectra)
```

The output of the ```fit_transform()``` method is now a ```pandas.DataFrame``` object. 

{: .highlight }
> Notice that by default the indices and the columns of the input data are not maintained to the output, and the ```spectra_airpls``` DataFrame has default indices and columns (see example below).

The ```spectra_airpls``` DataFrame has the following structure:

<iframe src="figures/processed.html" width="100%" style="border: none;"></iframe>

#### __Example 2: Using the ```set_output()``` API with a pipeline__

Similarly, the ```set_output()``` API can be used with pipelines. The following code shows how to create a pipeline that performs:

- Multiplicative scatter correction
- Standard scaling

```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.scatter import MultiplicativeScatterCorrection


pipeline = make_pipeline(MultiplicativeScatterCorrection(),StandardScaler())
pipeline.set_output(transform="pandas")

output = pipeline.fit_transform(spectra)
```
