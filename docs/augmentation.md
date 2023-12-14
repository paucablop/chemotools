---
title: Augmentation
layout: default
parent: Docs
---

# __Augmentation__

Data augmentation is a mathematical transformation of the spectral data that adds stochastic artifacts that resemble the ones that can be found in real-world data. The objective of data augmentation is to increase the size of the dataset and to improve the generalization of the model. The following algorithms are available:

- [Augmentation with normal noise](#augmentation-with-normal-noise)
- [Augmentation with uniform noise](#augmentation-with-uniform-noise)
- [Augmentation with exponential noise](#augmentation-with-exponential-noise)
- [Baseline shift](#baseline-shift)
- [Peak shift](#peak-shift)
- [Scale spectrum](#scale-spectrum)

## __Augmentation with normal noise__
Augmentation with normal noise to the spectrum. Gaussian noise with mean 0 and standard deviation defined by the user is added to each data-point of the spectrum.

### __Arguments__

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```scale``` | Standard deviation of the normal distribution. | ```float``` | ```0.0``` |
| ```random_state``` | Seed for the random number generator. | ```int``` | ```None``` |

### __Usage Example__
    
```python
from chemotools.augmentation import NormalNoise

normal_noise = NormalNoise(scale=0.001)
augmented_spectra = normal_noise.fit_transform(spectra)
```

### __Plotting Example__

<iframe src="figures/normal_noise_augmentation.html" width="800px" height="400px" style="border: none;"></iframe>

## __Augmentation with uniform noise__
Augmentation with uniform noise to the spectrum. Uniform noise with minimum and maximum values defined by the user is added to each data-point of the spectrum.

### __Arguments__

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```min``` | Minimum value of the uniform distribution. | ```float``` | ```0.0``` |
| ```max``` | Maximum value of the uniform distribution. | ```float``` | ```0.0``` |
| ```random_state``` | Seed for the random number generator. | ```int``` | ```None``` |

### __Usage Example__
    
```python
from chemotools.augmentation import UniformNoise

uniform_noise = UniformNoise(min=-0.001, max=0.001)
augmented_spectra = uniform_noise.fit_transform(spectra)
```

### __Plotting Example__

<iframe src="figures/uniform_noise_augmentation.html" width="800px" height="400px" style="border: none;"></iframe>

## __Augmentation with exponential noise__
Augmentation of the spectra by adding noise following an exponential distribution with a given standard distribution.

### __Arguments__

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```scale``` | Standard deviation of the exponential distribution. | ```float``` | ```0.0``` |
| ```random_state``` | Seed for the random number generator. | ```int``` | ```None``` |

### __Usage Example__
    
```python
from chemotools.augmentation import ExponentialNoise

exponential_noise = ExponentialNoise(scale=0.001)
augmented_spectra = exponential_noise.fit_transform(spectra)
```

### __Plotting Example__

<iframe src="figures/exponential_noise_augmentation.html" width="800px" height="400px" style="border: none;"></iframe>


## __Augmentation with baseline shift__
Adds a baseline to the data. The baseline is drawn from a one-sided uniform distribution between 0 and 0 + scale.

### __Arguments__

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```scale``` | Baseline to add. The baseline will be drawn from a unifrom distribution between ```0``` and ```0 + scale```| ```float``` | ```0.0``` |
| ```random_state``` | Seed for the random number generator. | ```int``` | ```None``` |


### __Usage Example__
    
```python
from chemotools.augmentation import BaselineShift

baseline = BaselineShift(scale=0.05)
augmented_spectra = baseline.fit_transform(spectra)
```

### __Plotting Example__

<iframe src="figures/baseline_shift_augmentation.html" width="800px" height="400px" style="border: none;"></iframe>


## __Augmentation with peak shift noise__
Augmentation of the spectra by shifting the peak a defined number of indices along the x-axis. This augmentation technique is specially interesting in Raman spectra, where peak shifts between or within instruments can occur as result of differences in the gratings. 

### __Arguments__

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```shift``` | Number of indices to the left and right that the spectra will be shifted. The method will select a random value between ```-shift``` to ```+shift``` following a uniform distribution.| ```float``` | ```0.0``` |
| ```random_state``` | Seed for the random number generator. | ```int``` | ```None``` |


### __Usage Example__
    
```python
from chemotools.augmentation import IndexShift

shift = IndexShift(shift=2)
augmented_spectra = shift.fit_transform(spectra)
```

### __Plotting Example__

<iframe src="figures/spectral_shift_augmentation.html" width="800px" height="400px" style="border: none;"></iframe>


## __Augmentation with peak scaling__
Scales the data by a value drawn from the uniform distribution centered around 1.0 with 

### __Arguments__

| Argument | Description | Type | Default |
| --- | --- | --- | --- |
| ```scale``` |     Range of the uniform distribution to draw the scaling factor from ```1 - shift``` to ```1 + shift``` following a uniform distribution.| ```float``` | ```0.0``` |
| ```random_state``` | Seed for the random number generator. | ```int``` | ```None``` |


### __Usage Example__
    
```python
from chemotools.augmentation import SpectrumScale

scale = SpectrumScale(scale=0.01)
augmented_spectra = scale.fit_transform(spectra)
```

### __Plotting Example__

<iframe src="figures/spectrum_scale_augmentation.html" width="800px" height="400px" style="border: none;"></iframe>
