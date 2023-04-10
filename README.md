# Welcome to chemotools! the project where chemometric tools are integrated with ```scikit-learn```

## __Introduction__

## __Scatter__

### __Multiplicative scatter correction__
Multiplicative scatter correction is a preprocessing technique in spectroscopy that corrects for the influence of light scattering on spectral measurements by dividing each spectrum by a scatter reference spectrum.

```python
from chemotools.scattering import MultiplicativeScatterCorrection

msc = MultiplicativeScatterCorrection()
spectra_msc = msc.fit_transform(spectra)
```

