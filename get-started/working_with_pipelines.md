---
title: Working with pipelines
layout: default
parent: Get Started
nav_order: 2
---


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