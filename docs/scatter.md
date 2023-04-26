---
title: Scatter
layout: default
parent: Docs
---

# __Scatter__

This package contains three common algorithms for scatter correction in spectroscopy:

- [Multiplicative scatter correction (MSC)](#multiplicative-scatter-correction)
- [Standard normal variate (SNV)](#standard-normal-variate)
- [Extended multiplicative scatter correction (EMSC)](#extended-multiplicative-scatter-correction)

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

## __Extended multiplicative scatter correction__

Coming soon
{: .label .label-yellow }

Extended multiplicative scatter correction (EMSC) is a preprocessing technique in spectroscopy that corrects for the influence of light scattering and instrumental drift by fitting a mathematical model to a reference spectrum and using it to normalize all spectra in the dataset.

An implementation of the EMSC will be available soon 🤓.