---
title: Scatter
layout: default
parent: Docs
---

# __Scatter__

This package contains three common algorithms for scatter correction in spectroscopy:

- [Multiplicative scatter correction (MSC)](#multiplicative-scatter-correction)
- [Extended multiplicative scatter correction (EMSC)](#extended-multiplicative-scatter-correction)
- [Standard normal variate (SNV)](#standard-normal-variate)
- [Robust normal variate (RNV)](#robust-normal-variate)

## __Multiplicative scatter correction__
Multiplicative scatter correction (MSC) is a preprocessing technique in spectroscopy that corrects for the influence of light scattering on spectral measurements by dividing each spectrum by a scatter reference spectrum. The current implementation, accepts three types of reference spectra:

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
|```use_mean``` | Whether to use the mean spectrum of the dataset as a reference spectrum. | ```bool``` | ```True``` |
| ```use_median``` | Whether to use the median spectrum of the dataset as a reference spectrum. | ```bool``` | ```False``` |
| ```reference``` | The reference spectrum to use for scatter correction. | ```numpy.ndarray``` | ```None``` |

### __Usage examples__:

#### __Case 1: usage example for the mean spectrum:__

```python
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection()
spectra_msc = msc.fit_transform(spectra)
``` 

#### __Case 2: usage example for the median spectrum:__

```python
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection(use_median=True)
spectra_msc = msc.fit_transform(spectra)
``` 

#### __Case 3: usage example for a single reference spectrum:__

```python
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection(reference=reference_spectrum)
spectra_msc = msc.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/multiplicative_signal_correction.html" width="800px" height="400px" style="border: none;"></iframe>

## __Extended multiplicative scatter correction__

Extended multiplicative scatter correction (EMSC) is a preprocessing technique for removing non linear scatter effects from spectra. It is based on fitting a polynomial regression model to the spectrum using a reference spectrum. The reference spectrum can be the mean or median spectrum of a set of spectra or a selected reference. The current implementation is based on the following articles:

- Nils Kristian Afseth, Achim Kohler. Extended multiplicative signal correction in vibrational spectroscopy, a tutorial, doi:10.1016/j.chemolab.2012.03.004

- Valeria Tafintseva et al. Correcting replicate variation in spectroscopic data by machine learning and model-based pre-processing, doi:10.1016/j.chemolab.2021.104350

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
|```use_mean``` | Whether to use the mean spectrum of the dataset as a reference spectrum. | ```bool``` | ```True``` |
| ```use_median``` | Whether to use the median spectrum of the dataset as a reference spectrum. | ```bool``` | ```False``` |
| ```reference``` | The reference spectrum to use for scatter correction. | ```numpy.ndarray``` | ```None``` |
| ```order``` | The degree of the polynomial regression model. | ```int``` | ```2``` |
| ```weights``` | The weights to use for the polynomial regression model. If ```None``` all weights will be set to 1.| ```numpy.ndarray``` | ```None``` |

### __Usage examples__:

```python
from chemotools.scatter import ExtendedMultiplicativeScatterCorrection

emsc = ExtendedMultiplicativeScatterCorrection()
spectra_emsc = emsc.fit_transform(spectra)
``` 

### __Plotting example__:

<iframe src="figures/extended_multiplicative_signal_correction.html" width="800px" height="400px" style="border: none;"></iframe>


## __Standard normal variate__
Standard normal variate (SNV) is a preprocessing technique in spectroscopy that adjusts for baseline shifts and variations in signal intensity by subtracting the mean and dividing by the standard deviation of each spectrum.

### __Arguments__:

The current implementation does not require any arguments.

### __Usage example__:

```python
from chemotools.scatter import StandardNormalVariate

snv = StandardNormalVariate()
spectra_snv = snv.fit_transform(spectra)
```

### __Plotting example__:
<iframe src="figures/standard_normal_variate.html" width="800px" height="400px" style="border: none;"></iframe>


## __Robust normal variate__
Robust normal variate (RNV) is a preprocessing technique in spectroscopy that adjusts for baseline shifts and variations in signal intensity. In contrast to the standard normal variate, the robust normal variate uses the mean and the standard deviation of a certain percentile of the spectrum to ensure robustness against outliers. The current implementation is based on:

- Q. Guo, W. Wu, D.L. Massart. The robust normal variate transform for pattern recognition with near-infrared data. doi:10.1016/S0003-2670(98)00737-5

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
|```percentile``` | The percentile of the spectrum to use for calculating the mean and standard deviation. | ```float``` | ```25``` |

### __Usage example__:

```python
from chemotools.scatter import RobustNormalVariate

rnv = RobustNormalVariate()
spectra_rnv = rnv.fit_transform(spectra)
```

### __Plotting example__:
<iframe src="figures/robust_normal_variate.html" width="800px" height="400px" style="border: none;"></iframe>

