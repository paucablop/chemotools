# Welcome to chemotools! 

the project where chemometric tools are integrated with ```scikit-learn```

## __Introduction__

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
from chemotools.scattering import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection()
spectra_msc = msc.fit_transform(spectra)
``` 

Usage example for the median spectrum:

```python
from chemotools.scattering import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection(use_median=True)
spectra_msc = msc.fit_transform(spectra)
``` 

Usage example for a single reference spectrum:

```python
from chemotools.scattering import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection(reference=reference_spectrum)
spectra_msc = msc.fit_transform(spectra)
```

![msc](figures/msc.png)


### __Standard normal variate__
Standard normal variate (SNV) is a preprocessing technique in spectroscopy that adjusts for baseline shifts and variations in signal intensity by subtracting the mean and dividing by the standard deviation of each spectrum.

Usage example for a single reference spectrum:

```python
from chemotools.scattering import StandardNormalVariate

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

- ```window_size```: The length of the window. Must be an odd integer number. _Default: 5_.
- ```polynomial_order```: The order of the polynomial used to fit the samples. Must be less than ```window_size```. _Default: 2_.
- ```derivative_order```: The order of the derivative to compute. _Default: 1_.
- ```mode```: The mode of the boundary. _Default: 'nearest'_, available options: ```'nearest'```, ```'constant'```, ```'reflect'```, ```'wrap'```, ```'mirror'```, ```'interp'```. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html for more information.

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

- ```window_size```: The length of the window. Must be an odd integer number. _Default: 5_.
- ```gap_size```: The number of points between the first and second points of the window. _Default: 3_.
- ```derivative_order```: The order of the derivative to compute. _Default: 1_.
- ```mode```: The mode of the boundary. _Default: 'nearest'_, available options: ```‘reflect’```, ```‘constant’```, ```‘nearest’```, ```‘mirror’```, ```‘wrap’```. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html for more information.

Usage example:

```python
from chemotools.derivative import NorrisWilliams

nw = NorrisWilliams(window_size=15, gap_size=3, derivative_order=1)
spectra_derivative =   nw.fit_transform(spectra)
```
![sgd](figures/sgd.png)
