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

- Ethiopia 🇪🇹
- Brasil 🇧🇷
- Spain 🇪🇸 (grown in a greenhouse!)

The spectra are measured from already brewed coffees using attenuated total refractance mid infrared spectroscopy (ATR-MIR). 

{: .highlight }
> __Geeky note:__ Yes! I know, if you are a coffee lover you will realize that there can be many factors affecting the spectra - roasting and brewing style, the coffee concentration will be different on a dark roasted espresso than on a light roasted poureover. In this dataset, all coffees were roasted under the same conditions, but the Ethiopian and Brazilian coffees were brewed for espresso and the Spanish using a moka pot.







