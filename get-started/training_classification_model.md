---
title: Training a classification model
layout: default
parent: Get Started
nav_order: 4
---

# __Training a classification model__

This page shows how to use ```chemotools``` and ```scikit-learn``` to train a partial least squares discriminant analysis (PLS-DA) classification model. A comprehensive explanation of PLS-DA can be found in the [Wikipedia page](https://en.wikipedia.org/wiki/Partial_least_squares_regression)

- [The coffee dataset](#the-coffee-dataset-☕)
- [Importing the data](#importing-the-data)
- [Plot, plot, plot and color](#plot-plot-plot-and-color)
- [Exploring the data](#exploring-the-data-🤓)
- [Preprocessing the spectra](#preprocessin-the-spectra)
- [Modelling the data](#modelling-the-data)


## __The coffee dataset ☕__

Can coffees from different origins be differentiated using infrared spectroscopy (IR)? Let's find out! This data set contains IR spectra of coffee with three origins:

- 🇪🇹-Ethiopia
- 🇧🇷-Brasil
- 🇪🇸-Spain (grown in a greenhouse!)

The spectra are measured from already brewed coffees using attenuated total refractance mid infrared spectroscopy (ATR-MIR). 

{: .highlight }
> Yes! I know. If you are a coffee lover you will be thinking: _there can be many factors affecting the spectra: an espresso is very different than a pourover!_. You are right! in this dataset, all coffees were roasted under the same conditions, but the Ethiopian and Brazilian coffees were brewed for espresso and the Spanish using a moka pot.

## __Importing the data__


Great! now that we know the context of the data, let's dive into it. I have loaded the data sets into a ```pandas.DataDrame```:


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

<iframe src="figures/origins_pie.html" width="800px" height="500px" style="border: none;"></iframe>

This is a balanced dataset where all the classes are equally represented 🤩!

## __Plot, plot, plot and color__

Plotting and visualizing the spectra is key to understand the data. In this case, we will plot and color the spectra according to their origin:

<iframe src="figures/coffee_data.html" width="800px" height="500px" style="border: none;"></iframe>

By plotting and coloring the spectra according to the origin, we can visually distinguish the Spanish coffee from the Ethiopian and the Brazilian.

## __Exploring the data 🤓__

Before starting with the classification model, we can have a look at the raw data using principal component analysis (PCA). To do so, we will mean center the data using the ```StandardScaler()``` preprocessing from ```scikit-learn```. Then, we factorize the preprocessed data into its principal components using the ```PCA()``` object from ```scikit-learn```.

{: .highlight }
> When using the ```StandardScaler()``` in spectroscopic models, we do not want to scale the standard deviation. This is why we set the attribute ```with_std``` to false. 

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_std=False)
pca =  PCA(n_components=2)

preprocessed_spectra = scaler.fit_transform(spectra)
scores = pca.fit_transform(preprocessed_spectra)
```

Let's look at the score plots for a two components model:

<iframe src="figures/coffee_pca.html" width="800px" height="500px" style="border: none;"></iframe>

The score plots reveals a clear separation of the spectra by coffee origin on the first component. The grouping in the second component corresponds to the different measuring days. 

## __Preprocessing the spectra__

The objective of the preprocessing is to remove from the spectra non-chemical systematic variation, such as baseline shifts or scattering effects. Here we will create a preprocessing [pipeline](https://paucablop.github.io/chemotools/get-started/scikit_learn_integration.html#working-with-pipelines) to combine ```chemotools``` and ```scikit-learn``` preprocessing algorithms.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chemotools.derivative import SavitzkyGolay
from chemotools.scatter import StandardNormalVariate
from chemotools.variable_selection import RangeCut

pipeline = make_pipeline(
    StandardNormalVariate(),
    SavitzkyGolay(window_size=21, polynomial_order=1),
    RangeCut(start=10, end=1350),
    StandardScaler(with_std=False))

preprocessed_spectra = pipeline.fit_transform(spectra)
```
This preprocessing pipeline contains 4 steps. The preprocessed spectra are shown in the image below.

<iframe src="figures/coffee_preprocessed_data.html" width="800px" height="500px" style="border: none;"></iframe>


## __Modelling the data__

Finally, lets model the data!! To make a classification using PLS-DA we need to encode our categorical variables (origins) into a numerical format:

| __Origin__  | __Encoded variable__ |
|-------------|:--------------------:|
| 🇪🇹-Ethiopia |          -1          |
| 🇧🇷-Brasil   |           0          |
| 🇪🇸-Spain    |           1          |

To do so, we can use the following function:

```python
def numerical_encoder(origin: str) -> int:
    if origin == '🇪🇹-Ethiopia':
        return -1
    
    if origin == '🇧🇷-Brasil':
        return 0

    if origin == '🇪🇸-Spain':
        return 1

encoded_variables = [numerical_encoder(origin) for origin in origins]
```

Great! Now we are almost ready for the PLS-DA modelling, but before we will do one more thing. It is good practice to split the data into training and testing splits, used to train and to evaluate the model respectively. To split the data, we can use super-cool [```train_test_split()```](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function form ```scikit-learn```.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_data, types, test_size=0.2, random_state=42)
```
And NOW we are ready to model the data. ``

```python
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=2)
pls.fit(X_train, y_train) # Train with train split

y_pred = pls.predict(X_test) # Test with test split
```
The PLS-DA algorithm will provide a continuous prediction of each sample, now we need to define the categorization criteria. For example, according to our encoding, a sample with a predicted value of 0.9 will be of Spanish origin, while a sample with a predicted value of -0.05 will be Brazilian.

```python
def categoriztion(prediction: float) -> int:
    if y < -0.5:
        return -1

    elif y < 0.5:
        return 0

    else:
        return 1

y_pred_categories = [categoriztion(prediction) for prediction in y_pred]
```

Cool, we have made the model, but... how does it perform? We can use some tools from ```scikit-learn``` to evaluate the performance of the classification model. In this case we will look at the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) and the accuracy.

```python
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy: ", accuracy_score(y_test, y_pred_categories))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred_categories))
```
which will print:

```python
>Accuracy: 1.0
>Confusion matrix: 
 [[7 0 0]
 [0 4 0]
 [0 0 9]]
```
from these results, can see that the classifier performs very well in the testing set. The confusion matrix can also be visualized as follows:

<iframe src="figures/confussion_matrix.html" width="800px" height="500px" style="border: none;"></iframe>






