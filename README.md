# Welcome to chemotools! 

the project where chemometric tools are integrated with ```scikit-learn```

## __Introduction__

## __Scatter__

### __Multiplicative scatter correction__
Multiplicative scatter correction is a preprocessing technique in spectroscopy that corrects for the influence of light scattering on spectral measurements by dividing each spectrum by a scatter reference spectrum. The current implementation, accepts three types of reference spectra:

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

![alt text](figures/msc.png)




