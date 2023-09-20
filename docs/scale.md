---
title: Scale
layout: default
parent: Docs
---

# __Scale__
Scale is a preprocessing technique in spectroscopy that scales the spectra. The following algorithms are available:
- [Index scaler](#index-scaler)
- [MinMax scaler](#minmax-scaler)
- [Norm scaler](#l-norm-scaler)


## __Index scaler__
Index scaler is a preprocessing technique in spectroscopy that scales each spectrum by the absorbance/intensity at a given index. Note that the index is not the wavenumber, but the index of the spectrum.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```index``` | The index of the spectrum to use for scaling. | ```int``` | ```0``` |

### __Usage examples__:

```python
from chemotools.scale import IndexScaler

index = IndexScaler(index=310)
spectra_norm = index.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/index_scaler.html" width="800px" height="400px" style="border: none;"></iframe>

## __MinMax scaler__
MinMaxScaler is a preprocessing technique in spectroscopy subtracts the minimum value of the spectrum and divides it by the difference between the maximum and the minimum value of the spectrum. If the parameter ```use_min``` is set to ```False```, the spectrum is just divided by the maximum value of the spectrum.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```use_min``` | If ```True```, the spectrum is subtracted by its minimum value and divided by the difference between the maximum and the minimum. If ```False```, the spectrum is scaled by its maximum value. | ```bool``` | ```True``` |

### __Usage examples__:

```python
from chemotools.scale import MinMaxScaler

minmax = MinMaxScaler()
spectra_norm = minmax.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/min_max_normalization.html" width="800px" height="400px" style="border: none;"></iframe>


## __L-norm scaler__
L-normalization is a preprocessing technique in spectroscopy that scales each spectrum by its L-norm. 

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```l_norm``` | The L-norm to use. | ```int``` | ```2``` |

### __Usage examples__:

```python
from chemotools.scale import NormScaler

lnorm = NormScaler(l_norm=2)
spectra_norm = lnorm.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/l_norm_scaler.html" width="800px" height="400px" style="border: none;"></iframe>

