# Welcome to chemotools! 

the project where chemometric tools are integrated with ```scikit-learn```

Table of contents
=================

<!--ts-->
   * [Installation](#installation)
   * [Scatter](#scatter)
        * [Multiplicative scatter correction](#multiplicative-scatter-correction)
        * [Standard normal variate](#standard-normal-variate)
        * [Extended multiplicative scatter correction](#extended-multiplicative-scatter-correction)
   * [Derivatives](#derivatives)
        * [Savitzky-Golay derivative](#savitzky-golay-derivative)
        * [William Norris derivative](#william-norris-derivative)
   * [Baseline](#baseline)
        * [Linear baseline correction](#linear-baseline-correction)

<!--te-->


## __Installation__
This package is available on PyPI and can be installed using pip:

```bash
pip install chemotools
```

## __Scatter__

This package contains three common algorithms for scatter correction in spectroscopy:

- Multiplicative scatter correction (MSC)
- Standard normal variate (SNV)
- Extended multiplicative scatter correction (EMSC)

### __Multiplicative scatter correction__
Multiplicative scatter correction (MSC) is a preprocessing technique in spectroscopy that corrects for the influence of light scattering on spectral measurements by dividing each spectrum by a scatter reference spectrum. The current implementation, accepts three types of reference spectra:

- The mean spectrum of the dataset (_default_).
- The median spectrum of the dataset.
- A single spectrum that is used to correct all spectra in the dataset.

Usage example for a single reference spectrum:

Usage example for the mean spectrum:

```python
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection()
spectra_msc = msc.fit_transform(spectra)
``` 

Usage example for the median spectrum:

```python
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection(use_median=True)
spectra_msc = msc.fit_transform(spectra)
``` 

Usage example for a single reference spectrum:

```python
from chemotools.scatter import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection(reference=reference_spectrum)
spectra_msc = msc.fit_transform(spectra)
```

![msc](figures/msc.png)


### __Standard normal variate__
Standard normal variate (SNV) is a preprocessing technique in spectroscopy that adjusts for baseline shifts and variations in signal intensity by subtracting the mean and dividing by the standard deviation of each spectrum.

Usage example for a single reference spectrum:

```python
from chemotools.scatter import StandardNormalVariate

snv = StandardNormalVariate()
spectra_snv = snv.fit_transform(spectra)
```
![snv](figures/snv.png)


### __Extended multiplicative scatter correction__
Extended multiplicative scatter correction (EMSC) is a preprocessing technique in spectroscopy that corrects for the influence of light scattering and instrumental drift by fitting a mathematical model to a reference spectrum and using it to normalize all spectra in the dataset.

An implementation of the EMSC will be available soon 🤓.

## __Derivatives__

This package contains two common algorithms for calculating derivatives in spectroscopy:

- Savitzky-Golay derivative
- William Norris derivative

### __Savitzky-Golay derivative__
Savitzky-Golay derivative is a preprocessing technique in spectroscopy that calculates the derivative of a spectrum by fitting a polynomial to a window of adjacent points and calculating the derivative of the polynomial.

The following arguments can be set:

- ```window_size: int```: The length of the window. Must be an odd integer number. _Default: 5_.
- ```polynomial_order: int```: The order of the polynomial used to fit the samples. Must be less than ```window_size```. _Default: 2_.
- ```derivative_order: int```: The order of the derivative to compute. _Default: 1_.
- ```mode: str```: The mode of the boundary. _Default: 'nearest'_, available options: ```'nearest'```, ```'constant'```, ```'reflect'```, ```'wrap'```, ```'mirror'```, ```'interp'```. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html for more information.

Usage example:

```python
from chemotools.derivative import SavitzkyGolay

sg = SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order=1)
spectra_derivative = sg.fit_transform(spectra)
```

![sgd](figures/sgd.png)

### __William Norris derivative__
William Norris derivative is a preprocessing technique in spectroscopy that calculates the derivative of a spectrum using finite differences.

The following arguments can be set:

- ```window_size: int```: The length of the window. Must be an odd integer number. _Default: 5_.
- ```gap_size: int```: The number of points between the first and second points of the window. _Default: 3_.
- ```derivative_order: int```: The order of the derivative to compute. _Default: 1_.
- ```mode: str```: The mode of the boundary. _Default: 'nearest'_, available options: ```‘reflect’```, ```‘constant’```, ```‘nearest’```, ```‘mirror’```, ```‘wrap’```. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html for more information.

Usage example:

```python
from chemotools.derivative import NorrisWilliams

nw = NorrisWilliams(window_size=15, gap_size=3, derivative_order=1)
spectra_derivative =   nw.fit_transform(spectra)
```
![wn](figures/wn.png)

## __Baseline__
Baseline correction is a preprocessing technique in spectroscopy that corrects for baseline shifts and variations in signal intensity by subtracting a baseline from a spectrum. The following algorithms are available:

- Linear baseline correction
- Polynomial baseline correction
- Cubic spline baseline correction
- Alternate iterative reweighed penalized least squares (AIRPLS) baseline correction
- Non-negative

### __Linear baseline correction__
Linear baseline correction is a preprocessing technique in spectroscopy that corrects for baseline shifts and variations in signal intensity by subtracting a linear baseline from a spectrum. The current implementation subtracts a linear baseline between the first and last point of the spectrum.

Usage example:

```python
from chemotools.baseline import LinearCorrection

lc = LinearCorrection()
spectra_baseline = lc.fit_transform(spectra)
```
![lb](figures/lb.png)


### __Polynomial baseline correction__
Polynomial baseline correction is a preprocessing technique in spectroscopy that approximates a baseline by fitting a polynomial to selected points of the spectrum. The selected points often correspond to minima in the spectra, and are selected by their index (not by the wavenumber). If no points are selected, the algorithm will select the first and last point of the spectrum.

The following arguments can be set:

- ```order: int``` The order of the polynomial used to fit the samples. _Default: 1_.
- ```indices: tuple``` The indices of the points to use for fitting the polynomial. _Default: (0, -1)_. At the moment the indices need to be specified manually as a tuple because ```scikit-learn``` does not support mutable attributes in ```BaseEstimator```. This tuple is transformed to a list when the ```transform``` method is called.

Usage example:

```python
from chemotools.baseline import PolynomialCorrection

pc = PolynomialCorrection(order=2, indices=(0, 75, 150, 200, 337))
spectra_baseline = pc.fit_transform(spectra)
```
![pb](figures/pb.png)

### __Cubic spline baseline correction__
Cubic spline baseline correction is a preprocessing technique in spectroscopy that approximates a baseline by fitting a cubic spline to selected points of the spectrum. Similar to the ```PolynomialCorrection```, the selected points often correspond to minima in the spectra, and are selected by their index (not by the wavenumber). If no points are selected, the algorithm will select the first and last point of the spectrum. 

The following arguments can be set:
- ```indices: tuple``` The indices of the points to use for fitting the polynomial. _Default: None_. At the moment the indices need to be specified manually as a tuple because ```scikit-learn``` does not support mutable attributes in ```BaseEstimator```. This tuple is transformed to a list when the ```transform``` method is called.

Usage example:

```python
from chemotools.baseline import CubicSplineCorrection

cspl = CubicSplineCorrection(indices=(0, 75, 150, 200, 337))
spectra_baseline = cspl.fit_transform(spectra)
```

![splines](figures/splines.png)

### __Alternate iterative reweighed penalized least squares (AIRPLS) baseline correction__
It is an automated baseline correction algorithm that uses a penalized least squares approach to fit a baseline to a spectrum. The original algorithm is based on the paper by [Zhang et al.](https://pubs.rsc.org/is/content/articlelanding/2010/an/b922045c). The current implementation is based on the Python implementation by [zmzhang](https://github.com/zmzhang/airPLS).

The following arguments can be set:
- ```nr_iterations: int``` The number of iterations before exiting the algorithm. _Default: 15_.
- ```lam: float``` smoothing factor. _Default: 1e2_.
- ```polynomial_order: int``` The order of the polynomial used to fit the samples. _Default: 1_.

Usage example:

```python
from chemotools.baseline import AirPls

airpls = AirPls()
spectra_baseline = airpls.fit_transform(spectra)
```

![airpls](figures/airpls.png)

### __Non-negative__
Non-negative baseline correction is a preprocessing technique in spectroscopy that corrects for baseline by removing negative values from a spectrum. Negative values are either replaced by 0, or set to their absolute value.

The following arguments can be set:
- ```mode: str``` If ```'zero'```, negative values are replaced by 0. If ```'abs'```, negative values are set to their absolute value. _Default: ```'zero'```.

Usage example:

```python
from chemotools.baseline import NonNegative

nnz = NonNegative(mode='zero')
nna = NonNegative(mode='abs')
spectra_nnz = nnz.fit_transform(spectra_baseline)
spectra_nna = nna.fit_transform(spectra_baseline)
```

![nnz](figures/nnz.png)
![nna](figures/nna.png)

## __Normalization__
