---
title: Variable selection
layout: default
parent: Docs
---

# __Variable selection__
Variable selection is a preprocessing technique in spectroscopy that selects the most relevant variables. The following algorithms are available:
- [Range cut by index](#range-cut-by-index)
- [Range cut by wavenumber](#range-cut-by-wavenumber)

## __Range cut by index__
Range cut by index is a preprocessing technique in spectroscopy that selects all the variables in the spectra given a range of two indices.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```start``` | The index of the first variable to select. | ```int``` | ```0``` |
| ```end``` | The index of the last variable to select. | ```int``` | ```-1``` |

### __Usage examples__:

```python
from chemotools.variable_selection import RangeCutByIndex

rcbi = RangeCutByIndex(10, 200)
spectra_rcbi = rcbi.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/range_cut_by_index.html" width="800px" height="400px" style="border: none;"></iframe>


## __Range cut by wavenumber__
Range cut by wavenumber is a preprocessing technique in spectroscopy that selects all the variables in the spectra given a range of two wavenumbers. 

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```wavenumbers```| The wavenumbers of the spectra. |```numpy.ndarray```/```list```| ```None``` |
| ```start``` | The wavenumber of the first variable to select. | ```float``` | ```0``` |
| ```end``` | The wavenumber of the last variable to select. | ```float``` | ```-1``` |

{: .note }
>The ```RangeCutByWavenumber()``` will store the indices of the selected variables. This is useful when the same range of wavenumbers is used to select variables in different spectra.

{: .warning }
>The ```wavenumbers``` vector myst be sorted in ascending order.



### __Usage examples__:

```python
from chemotools.variable_selection import RangeCutByWavenumber

rcbw = RangeCutByWavenumber(wn, 950, 1100)
spectra_rcbw = rcbw.fit_transform(spectra)
```
### __Plotting example__:

<iframe src="figures/range_cut_by_wavenumber.html" width="800px" height="400px" style="border: none;"></iframe>
