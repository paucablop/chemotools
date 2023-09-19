---
title: Variable selection
layout: default
parent: Docs
---

# __Variable selection__
Variable selection is a preprocessing technique in spectroscopy that selects the most relevant variables. The following algorithms are available:
- [Range cut](#range-cut)
- [SelectFeatures](#range-cut-by-wavenumber)

## __Range cut__
Range cut by index is a preprocessing technique in spectroscopy that selects all the variables in the spectra given a range of either two indices or two wavenumbers.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```start``` | If not ```wavenumbers```, start corresponds to the first index. If the ```wavenumbers``` are provided, then it correpsonds to the first wavenumber. | ```float``` | ```0``` |
| ```end``` | If not ```wavenumbers```, end corresponds to the last index. If the ```wavenumbers``` are provided, then it correpsonds to the last wavenumber. | ```float``` | ```-1``` |
| ```wavenumbers```| The wavenumbers of the spectra. |```numpy.ndarray```/```list```| ```None``` |
    
{: .warning }
> The ```wavenumbers``` vector must be sorted in ascending order.

### __Usage examples__:

#### __Case 1: Range cut by index__

```python
from chemotools.variable_selection import RangeCut

rcbi = RangeCut(0, 200)
spectra_rcbi = rcbi.fit_transform(spectra)
```

#### __Case 2: Range cut by wavenumbers__


```python
from chemotools.variable_selection import RangeCut

rcbw = RangeCut(950, 1100, wavenumbers=wn)
spectra_rcbw = rcbw.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/range_cut_by_index.html" width="800px" height="400px" style="border: none;"></iframe>

<iframe src="figures/range_cut_by_wavenumber.html" width="800px" height="400px" style="border: none;"></iframe>

## __SelectFeatures__

Coming soon
{: .label .label-yellow }

This preprocessing technique allows selecting the relevant features in a spectra. 

