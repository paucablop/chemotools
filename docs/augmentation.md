---
title: Augmentation
layout: default
parent: Docs
---

# __Augmentation__

Data augmentation is a mathematical transformation of the spectral data that adds stochastic artifacts that resemble the ones that can be found in real-world data. The objective of data augmentation is to increase the size of the dataset and to improve the generalization of the model. The following algorithms are available:

- [Augmentation with normal noise](#augmentation with-normal-noise)
- [Augmentation with uniform noise](#augmentation with-uniform-noise)
- [Augmentation with exponential noise](#augmentation with-exponential-noise)
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

nn = NormalNoise(scale=0.001)
spectra_noise = nn.fit_transform(spectra)
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

un = UniformNoise(min=-0.001, max=0.001)
spectra_noise = un.fit_transform(spectra)
```

### __Plotting Example__

<iframe src="figures/uniform_noise_augmentation.html" width="800px" height="400px" style="border: none;"></iframe>

