---
title: Derivative
layout: default
parent: Docs
---

# __Derivatives__

This package contains two common algorithms for calculating derivatives in spectroscopy:

- [Savitzky-Golay derivative](#savitzky-golay-derivative)
- [William Norris derivative](#william-norris-derivative)

## __Savitzky-Golay derivative__
Savitzky-Golay derivative is a preprocessing technique in spectroscopy that calculates the derivative of a spectrum by fitting a polynomial to a window of adjacent points and calculating the derivative of the polynomial.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```window_size``` | The length of the window. Must be an odd integer number. | ```int``` | ```5``` |
| ```polynomial_order``` | The order of the polynomial used to fit the samples. Must be less than ```window_size```. | ```int``` | ```2``` |
| ```derivative_order``` | The order of the derivative to compute. | ```int``` | ```1``` |
| ```mode``` | The mode of the boundary. Options are ```'nearest'```, ```'constant'```, ```'reflect'```, ```'wrap'```, ```'mirror'```, ```'interp'```. See the [```scipy.savgol_filter```](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) for more information.| ```str``` | ```'nearest'``` |

### __Usage example__:

```python
from chemotools.derivative import SavitzkyGolay

sg = SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order=1)
spectra_derivative = sg.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/savitzky_golay_derivative.html" width="800px" height="400px" style="border: none;"></iframe>


## __William Norris derivative__
William Norris derivative is a preprocessing technique in spectroscopy that calculates the derivative of a spectrum using finite differences.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```window_size``` | The length of the window. Must be an odd integer number. | ```int``` | ```5``` |
|```gap_size``` | The number of points between the first and second points of the window. | ```int``` | ```3``` |
| ```derivative_order``` | The order of the derivative to compute. | ```int``` | ```1``` |
| ```mode``` | The mode of the boundary. Options are ```'nearest'```, ```'constant'```, ```'reflect'```, ```'wrap'```, ```'mirror'```, ```'interp'```. See the [```scipy.ndimage.convolve```](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html) for more information.| ```str``` | ```'nearest'``` |

### __Usage example__:

```python
from chemotools.derivative import NorrisWilliams

nw = NorrisWilliams(window_size=15, gap_size=3, derivative_order=1)
spectra_derivative =   nw.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/norris_williams_derivative.html" width="800px" height="400px" style="border: none;"></iframe>