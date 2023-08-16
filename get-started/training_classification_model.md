---
title: Training a classification model
layout: default
parent: Get Started
nav_order: 4
---

# __Training a classification model__

This page shows how to use ```chemotools``` and ```scikit-learn``` to train a partial least squares discriminant analysis (PLS-DA) classification model. 


## __The coffee dataset ☕__

Can coffees from different origins be differentiated using infrared spectroscopy (IR)? Let's find out! This data set contains IR spectra of coffee with three origins:

- 🇪🇹-Ethiopia
- 🇧🇷-Brasil
- 🇪🇸-Spain (grown in a greenhouse!)

The spectra are measured from already brewed coffees using attenuated total refractance mid infrared spectroscopy (ATR-MIR). 

{: .highlight }
> Yes! I know. If you are a coffee lover you will be thinking _that there can be many factors affecting the spectra: an espresso is very different than a pourover!_. You are right! in this dataset, all coffees were roasted under the same conditions, but the Ethiopian and Brazilian coffees were brewed for espresso and the Spanish using a moka pot.

### __Import the data__


Great! now that we know the context of the data, let's dive into the data. I have loaded the data sets into a ```pandas.DataDrame```:


```python
import pandas as pd

spectra, origins = load_coffee_data()
```

The ```spectra``` variable is a ```pandas.DataFrame``` containing 128 samples (rows) and 1841 features (columns). This can be inspected by:

```python
spectra.shape

> (128, 1841)
```

The ```origins``` variable is a ```list``` with 128 elements corresponding to the origin of each sample. The amount of samples of each origin is visualized in the following plot:

<iframe src="figures/origins_pie.html" width="100%" style="border: none;"></iframe>

In this dataset, the coffee from Spain is over-represented.

### __Plot and color__

Plotting and visualizing the spectra is key to understand the data. In this case, we will plot and color the spectra according to their origin:

<iframe src="figures/coffee_data.html" width="100%" style="border: none;"></iframe>

Just by plotting the data we can easily differentiate the Spanish coffee from the other two. 






