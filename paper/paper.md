---
title: 'chemotools: A Python Package that Integrates Chemometrics and scikit-learn'
tags:
    - Python
    - Chemometrics
    - Spectroscopy
    - Machine Learning
    - scikit-learn

authors:
- name: Pau Cabaneros Lopez
  orcid: 0000-0003-2372-5082
  affiliation: 1

affiliations:
- name: Novo Nordisk A/S, Bagsvaerd, Denmark
  index: 1

date: 1 December 2023
bibliography: ./paper.bib
---


# Summary

```chemotools``` stands as a production-oriented versatile Python library, developed to provide a unified platform for advancing chemometric model development. Integrating spectral preprocessing methodologies with the ```scikit-learn``` API and the expansive Python machine learning ecosystem, this library seeks to standardize and simplify the complex process of creating and implementing robust chemometric and machine learning models of spectral data. 

# Statement of need

Spectroscopy is an analytical technique used to understand the composition of materials using light. Traditionally, spectroscopic data is analyzed by a discipline called chemometrics, a branch of machine learning specialized on extracting chemical information from multivariate spectra. Over the last decades, chemometricians, have excelled by developing advanced preprocessing methods designed to remove instrument and measuring artifacts from the spectra, isolating the pure chemical information of the samples [@RINNAN20091201], [@MISHRA2020116045]. 

Since spectroscopic methods are faster and simpler than most of other analytical techniques, their adoption as integral components of Process Analytical Technology (PAT) has witnessed significant growth across industries, including chemical, biotech, food, and pharmaceuticals. Despite this surge, a notable obstacle has been the absence of open-source standardized, accessible toolkit for chemometric model development and deployment. ```chemotools```, positioned as a comprehensive solution, addresses this void by integrating into the Python machine learning ecosystem. By implementing a variety of preprocessing and feature selection tools with the ```scikit-learn``` API [@pedregosa2018scikitlearn], ```chemotools``` opens up the entire ```scikit-learn``` toolbox to users, encompassing features such as:

- a rich collection of estimators for regression, classification, and clustering
- cross-validation and hyper-parameter optimization algorithms
- pipelining for efficient workflows
- and model persistence to standardized files such as ```joblib``` or ```pickle```

This integration empowers users with a versatile array of tools for robust model development and evaluation (\autoref{fig:1}).

In addition to its foundational capabilities, ```chemotools``` not only enables users to preprocess data and construct/train models using ```scikit-learn``` but also streamlines the transition of these models into a production setting. By enabling users with a well defined interface, ```chemotools``` facilitates the reception of input data and delivery of predictions from the trained model. This can then be containerized using Docker, providing an efficient means for the distribution and implementation of the model in any Docker-compatible environment, facilitating the deployment of models to cloud environments. This adaptive capability not only enables organizations to scale model usage but also allows them to monitor performance and promptly update or rollback the model as necessary.

In addition, ```chemotools``` introduces a practical innovation by providing a standardized framework for data augmentation of spectroscopic datasets through the ```scikit-learn``` API. This feature offers users a straightforward and consistent method to enhance their datasets, contributing to improved model generalization. By integrating data augmentation into the chemometric workflow, ```chemotools``` provides users with an efficient tool for refining their datasets and optimizing model performance. 


![chemotools workflow .\label{fig:1}](../assets/images/overview_2.png)


# Features and functionality

```chemotools``` implements a collection of ```scikit-learn``` transformers and selectors. Transformers are divided in preprocessing and augmentation methods. Preprocessing functions range from well-established chemometric methods such as multiplicative scatter correction or standard normal variate [@RINNAN20091201], to more recent methods such as asymmetrically reweighed penalized least squares baseline correction method used to remove complex baselines [@arpls2]. The augmentation module implements methods to add stochastic artifacts to the spectral data. These artifacts can range from adding noise following a given distribution to shifts on the spectral peaks or changes on the intensity of the peaks. Besides the transformers, ```chemotools``` also implements selectors, used to select the relevant features of the spectra that contain the chemical information. In addition to the mathematical methods, ```chemotools``` also provides real-world spectral datasets complemented [@cabaneros1] with an extensive documentation page of the different methods as well guides showcasing how to combine ```scikit-learn``` and ```chemotools``` to train regression and classification models (https://paucablop.github.io/chemotools/).

# Adoption in educational applications

Beyond its practical applications, ```chemotools``` has being utilized as an educational tool at universities for both Master's (MSc) and Doctoral (PhD) levels. Its incorporation into academic curricula provides a valuable way to enable the students to benefit from hands-on experience on real-world datasets gaining practical insights into the application of sophisticated techniques for preprocessing and analyzing spectral data. The tool's user-friendly interface, coupled with comprehensive documentation, has proven and enriching learning experience for students pursuing higher education in fields related to analytical chemistry and chemometrics.

# Author contribution statement

Conceptualization, coding, developing and paper writing by Pau Cabaneros Lopez.

# Acknowledgements

This project has not received any external funding.

# References


