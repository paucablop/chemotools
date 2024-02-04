---
title: Feature selection
layout: default
parent: Docs
---

# __Feature selection__
Feature selection is a preprocessing technique in spectroscopy that selects the most relevant features. The following algorithms are available:
- [Range cut](#range-cut)
- [IndexSelector](#index-selector)

{: .note }
> The variable selection algorithms implemented in ```chemotools``` allow you to select a subset of variables/features from the spectra. They are not designed to find the most relevant variables/features for a given task. 

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
from chemotools.feature_selection import RangeCut

rcbi = RangeCut(0, 200)
spectra_rcbi = rcbi.fit_transform(spectra)
```

#### __Case 2: Range cut by wavenumbers__


```python
from chemotools.feature_selection import RangeCut

rcbw = RangeCut(950, 1100, wavenumbers=wn)
spectra_rcbw = rcbw.fit_transform(spectra)
```

After fitting the method with the wavenumbers, the selected wavenumbers can be accessed using the ```wavenumbers_``` attribute.


### __Plotting example__:

<iframe src="figures/range_cut_by_index.html" width="800px" height="400px" style="border: none;"></iframe>

<iframe src="figures/range_cut_by_wavenumber.html" width="800px" height="400px" style="border: none;"></iframe>


## __Index selector__
IndexSelector is a preprocessing technique in spectroscopy that selects the most relevant variables. The selected features do not need to be continuous in the spectra, but they can be located at different locations. The algorithm allows selecting the features by imputing a list of indices or wavenumbers.

### __Arguments__:

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```features``` | The indices or wavenumbers of the features to be selected. If ```None``` it will return the entire array. | ```numpy.ndarray```/```list``` | ```None``` |
| ```wavenumbers```| The wavenumbers of the spectra. |```numpy.ndarray```/```list```| ```None``` |
    
{: .warning }
> The ```wavenumbers``` vector must be sorted in ascending order.

### __Usage examples__:

In the example below, the selected wavenumbers ```wn_select``` are used to select the features in the spectra.The selected wavenumbers include features from the beginning, middle and end of the spectra.


```python
from chemotools.feature_selection import IndexSelector

sfbw = IndexSelector(features=wn_select,wavenumbers=wn)
spectra_sfbw = sfbw.fit_transform(spectra)
```

### __Plotting example__:

<iframe src="figures/select_features_by_wavenumber.html" width="800px" height="400px" style="border: none;"></iframe>