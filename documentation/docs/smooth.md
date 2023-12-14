---
title: Smooth
layout: default
parent: Docs
---

# __Smooth__
Smooth is a preprocessing technique in spectroscopy that smooths the spectra. The following algorithms are available:
- [Savitzky-Golay filter](#savitzky-golay-filter)
- [Whittaker smoother](#whittaker-smoother)
- [Mean filter](#mean-filter)
- [Median filter](#median-filter)

## __Savitzky-Golay filter__
Savitzky-Golay filter is a preprocessing technique in spectroscopy that smooths the spectra by fitting a polynomial to the data. The current implementation is based on the ```scipy.signal.savgol_filter``` function.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```window_size``` | The length of the window. Must be an odd integer number. | ```int``` | ```3``` |
| ```polynomial_order``` | The order of the polynomial used to fit the samples. Must be less than ```window_size```. | ```int``` | ```1``` |
| ```derivative_order``` | The order of the derivative to compute. | ```int``` | ```1``` |
| ```mode``` | The mode of the boundary. Options are ```'nearest'```, ```'constant'```, ```'reflect'```, ```'wrap'```, ```'mirror'```, ```'interp'```. See the [```scipy.savgol_filter```](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) for more information.| ```str``` | ```'nearest'``` |

### __Usage examples__:

```python
from chemotools.smooth import SavitzkyGolayFilter

sgf = SavitzkyGolayFilter(window_size=15, polynomial_order=2)
spectra_norm = sgf.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/savitzky_golay_smoothing.html" width="800px" height="400px" style="border: none;"></iframe>


## __Whittaker smoother__
It is an automated smoothing algorithm that uses a penalized least squares approach to iteratively apply a smoothing operation to the data  by minimizing a penalty function that balances the degree of smoothness and the fidelity to the original data.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```lam``` | smoothing factor. | ```float``` | ```1e2``` |
| ```differences``` | The number of differences to use. | ```int``` | ```1``` |

### __Usage examples__:

```python
from chemotools.smooth import WhittakerSmooth

wtk = WhittakerSmooth(lam=10)
spectra_norm = wtk.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/whittaker_smoothing.html" width="800px" height="400px" style="border: none;"></iframe>

## __Mean filter__
Mean filter is a preprocessing technique in spectroscopy that smooths the spectra by applying a mean filter. The current implementation is based on the ```scipy.ndimage.uniform_filter``` function.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```window_size``` | The length of the window. Must be an odd integer number. | ```int``` | ```3``` |
| ```mode``` | The mode parameter determines how the array borders are handled, where ```'constant'```, ```'reflect'```, ```'wrap'```, ```'mirror'```, ```'interp'```. See the [official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html) for more information. | ```str``` | ```'nearest'``` |

### __Usage examples__:

```python
from chemotools.smooth import MeanFilter

mean_filter = MeanFilter()
spectra_norm = mean_filter.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/mean_smoothing.html" width="800px" height="400px" style="border: none;"></iframe>

## __Median filter__
Median filter is a preprocessing technique in spectroscopy that smooths the spectra by applying a median filter. The current implementation is based on the ```scipy.ndimage.median_filter``` function.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```window_size``` | The length of the window. Must be an odd integer number. | ```int``` | ```3``` |
| ```mode``` | The mode parameter determines how the array borders are handled, where ```'constant'```, ```'reflect'```, ```'wrap'```, ```'mirror'```, ```'interp'```. See the [official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html) for more information. | ```str``` | ```'nearest'``` |

### __Usage examples__:

```python
from chemotools.smooth import MedianFilter

median_filter = MedianFilter()
spectra_norm = median_filter.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/median_smoothing.html" width="800px" height="400px" style="border: none;"></iframe>


