---
title: Baseline
layout: default
parent: Docs
---

# __Baseline__
Baseline correction is a preprocessing technique in spectroscopy that corrects for baseline shifts and variations in signal intensity by subtracting a baseline from a spectrum. The following algorithms are available:

- [Linear baseline correction](#linear-baseline-correction)
- [Polynomial baseline correction](#polynomial-baseline-correction)
- [Cubic spline baseline correction](#cubic-spline-baseline-correction)
- [Alternate iterative reweighed penalized least squares baseline correction (AirPLS)](##alternate-iterative-reweighed-penalized-least-squares-baseline-correction-airpls)
- [Non-negative](#non-negative)
- [Subtract reference spectrum](#subtract-reference-spectrum)
- [Constant baseline correction](#constant-baseline-correction)

## __Linear baseline correction__
Linear baseline correction is a preprocessing technique in spectroscopy that corrects for baseline shifts and variations in signal intensity by subtracting a linear baseline from a spectrum. The current implementation subtracts a linear baseline between the first and last point of the spectrum.

### __Usage examples__:

```python
from chemotools.baseline import LinearCorrection

lc = LinearCorrection()
spectra_baseline = lc.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/linear_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>


## __Polynomial baseline correction__
Polynomial baseline correction is a preprocessing technique in spectroscopy that approximates a baseline by fitting a polynomial to selected points of the spectrum. The selected points often correspond to minima in the spectra, and are selected by their index (not by the wavenumber). If no points are selected, the algorithm will select all the points in the spectrum to fit a polynomial of a given order. This case is often called detrending in other spectral processing software.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```order``` | The order of the polynomial to fit. | ```int``` | ```1``` |
| ```indices``` | The indices of the points to use for fitting the polynomial. | ```list``` | ```None``` |


### __Usage examples__:

```python
from chemotools.baseline import PolynomialCorrection

pc = PolynomialCorrection(order=2, indices=[0, 75, 150, 200, 337])
spectra_baseline = pc.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/polynomial_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>

## __Cubic spline baseline correction__
Cubic spline baseline correction is a preprocessing technique in spectroscopy that approximates a baseline by fitting a cubic spline to selected points of the spectrum. Similar to the ```PolynomialCorrection```, the selected points often correspond to minima in the spectra, and are selected by their index (not by the wavenumber). If no points are selected, the algorithm will select the first and last point of the spectrum. 

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```indices``` | The indices of the points to use for fitting the polynomial. | ```list``` | ```None``` |


### __Usage examples__:

```python
from chemotools.baseline import CubicSplineCorrection

cspl = CubicSplineCorrection(indices=[0, 75, 150, 200, 337])
spectra_baseline = cspl.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/cubic_spline_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>


## __Alternate iterative reweighed penalized least squares baseline correction (AirPLS)__
It is an automated baseline correction algorithm that uses a penalized least squares approach to fit a baseline to a spectrum. The original algorithm is based on the paper by [Zhang et al.](https://pubs.rsc.org/is/content/articlelanding/2010/an/b922045c). The current implementation is based on the Python implementation by [zmzhang](https://github.com/zmzhang/airPLS).

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```nr_iterations``` | The number of iterations before exiting the algorithm. | ```int``` | ```15``` |
| ```lam``` | The smoothing factor. | ```float``` | ```1e2``` |
| ```polynomial_order``` | The order of the polynomial used to fit the samples. | ```int``` | ```1``` |

### __Usage examples__:

```python
from chemotools.baseline import AirPls

airpls = AirPls()
spectra_baseline = airpls.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/airpls_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>

## __Non-negative__
Non-negative baseline correction is a preprocessing technique in spectroscopy that corrects for baseline by removing negative values from a spectrum. Negative values are either replaced by 0, or set to their absolute value.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```mode``` | ```'zero'```, negative values are replaced by 0. ```'abs'```, negative values are set to their absolute value. | ```str``` | ```'zero'``` |

### __Usage example__:

```python
from chemotools.baseline import NonNegative

nnz = NonNegative(mode='zero')
nna = NonNegative(mode='abs')
spectra_nnz = nnz.fit_transform(spectra_baseline)
spectra_nna = nna.fit_transform(spectra_baseline)
```

### __Plotting example__:

<iframe src="figures/non_negative_zero_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>
<iframe src="figures/non_negative_abs_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>

## __Subtract reference spectrum__
Subtract reference spectrum is a preprocessing technique in spectroscopy that subtracts a reference spectrum from a target spectrum. The reference spectrum must be a single spectrum. The target spectrum can be a single spectrum or a list of spectra.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```reference``` | The reference spectrum. | ```numpyp.ndarray``` | ```None``` |

The following arguments can be set:
- ```reference: np.array``` The reference spectrum. _Default: None_. When it is set to None, the algorithm will not subtract the reference spectrum.

### __Usage example__:

```python
from chemotools.baseline import SubtractReference

sr = SubtractReference(reference=reference_spectrum)
spectra_sr = sr.fit_transform(spectra)
```
### __Plotting example__:

<iframe src="figures/subtract_reference_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>

## __Constant baseline correction__
Constant baseline correction is a preprocessing technique in spectroscopy that corrects for baseline by subtracting a constant value from a spectrum. The constant value is the mean of a region in the spectrum. This processing step is specially useful in UV-Vis spectroscopy, where there can be a large region in the spectra without any absorption.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
|```wavenumbers``` | The wavenumbers of the spectrum. | ```numpy.ndarray``` | ```None``` |
| ```start``` | The start index of the region to use for calculating the mean. If no wavenumbers are provided, it will take the index of the spectrum. If wavenumbers are provided it will take the index corresponding to the wavenumber | ```int``` | ```0``` |
| ```end``` | The end index of the region to use for calculating the mean. If no wavenumbers are provided, it will take the index of the spectrum. If wavenumbers are provided it will take the index corresponding to the wavenumber| ```int``` | ```1``` |

### __Usage example__:

#### __Case 1: No wavenumbers provided__

```python
from chemotools.baseline import ConstantBaselineCorrection

cbc = ConstantBaselineCorrection(start=0, end=30)
spectra_baseline = cbc.fit_transform(spectra)
```

#### __Case 2: Wavenumbers provided__

```python
from chemotools.baseline import ConstantBaselineCorrection

cbc = ConstantBaselineCorrection(wavenumbers=wn,start=950, end=975)
spectra_baseline = cbc.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/constant_baseline_correction.html" width="800px" height="400px" style="border: none;"></iframe>
