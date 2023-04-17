[![pypi](https://img.shields.io/pypi/v/chemotools)](https://pypi.org/project/chemotools)
[![pypi](https://img.shields.io/pypi/pyversions/chemotools)](https://pypi.org/project/chemotools)
[![pypi](https://img.shields.io/pypi/l/chemotools)](https://github.com/paucablop/chemotools/blob/main/LICENSE)
[![codecov](https://codecov.io/github/paucablop/chemotools/branch/main/graph/badge.svg?token=D7JUJM89LN)](https://codecov.io/github/paucablop/chemotools)

# __chemotools__

Welcome to Chemotools, a Python package that integrates chemometrics with Scikit-learn.

👉 Check the [documentation](https://paucablop.github.io/chemotools/) for a full description on how to use chemotools.

## Description

Chemotools is a Python package that provides a collection of preprocessing tools and utilities for working with spectral data. It is built on top of popular scientific libraries and is designed to be highly modular, easy to use, and compatible with Scikit-learn transformers.

If you are interested in learning more about chemotools, please visit the [documentation](https://paucablop.github.io/chemotools/) page.

Benefits:
- Provides a collection of preprocessing tools and utilities for working with spectral data
- Highly modular and compatible with Scikit-learn transformers
- Can perform popular preprocessing tasks such as baseline correction, smoothing, scaling, derivatization, and scattering correction
- Open source and available on PyPI

Applications:
- Analyzing and processing spectral data in chemistry, biology, and other fields
- Developing machine learning models for predicting properties or classifying samples based on spectral data
- Teaching and learning about chemometrics and data preprocessing in Python

## Installation

Chemotools is distributed via PyPI and can be easily installed using pip:

```bash
pip install chemotools
```

## Usage

Chemotools is designed to be used in conjunction with Scikit-learn. It follows the same API as other Scikit-learn transformers, so you can easily integrate it into your existing workflow. For example, you can use chemotools to build pipelines that include transformers from chemotools and Scikit-learn:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from chemotools.baseline import AirPls
from chemotools.scatter import MultiplicativeScatterCorrection

preprocessing = make_pipeline(AirPls(), MultiplicativeScatterCorrection(), StandardScaler(with_std=False)) 
spectra_transformed = preprocessing.fit_transform(spectra)
```

Check the [documentation](https://paucablop.github.io/chemotools/) for more information on how to use chemotools.


## Contributing

We welcome contributions to Chemotools from anyone interested in improving the package. Whether you have ideas for new features, bug reports, or just want to help improve the code, we appreciate your contributions! You are also welcome to see the [Project Board](https://github.com/users/paucablop/projects/4) to see what we are currently working on.

To contribute to Chemotools, please follow these guidelines:

#### Reporting Bugs

If you encounter a bug or unexpected behavior in Chemotools, please open an issue on the GitHub repository with a detailed description of the problem, including any error messages and steps to reproduce the issue. If possible, include sample code or data that demonstrates the problem.

#### Suggesting Enhancements

If you have an idea for a new feature or enhancement for Chemotools, please open an issue on the GitHub repository with a detailed description of the proposed feature and its benefits. If possible, include example code or use cases that illustrate how the feature would be used.

#### Submitting Changes

If you'd like to contribute code changes to Chemotools, please follow these steps:

- Create a new branch for your changes. We follow trunk-based development, so all changes should be made on a new branch and branches should be short-lived and merged into main.

- Write your code and tests, making sure to follow the Chemotools coding style and conventions. It is fundamental to include tests for both, the Scikit-learn API and the functionality of the transformers.

- Run the tests using the provided testing framework to ensure that your changes do not introduce any new errors or regressions.

- Submit a pull request to the main Chemotools repository with a detailed description of your changes and the problem they solve.

We will review your changes and provide feedback as soon as possible. If we request changes, please make them as quickly as possible to keep the review process moving.

#### Code Style

Please follow the Chemotools code style and conventions when contributing code changes. Specifically:

- Use four spaces for indentation
- Use descriptive variable names
- Avoid using magic numbers or hard-coded strings
- Format your code using Black

#### Codecov

We use Codecov to track the test coverage of Chemotools. Please make sure that your changes do not reduce the test coverage of the package.


## License

This package is distributed under the MIT license. See the [LICENSE](LICENSE) file for more information. When contributing code to Chemotools, you are agreeing to release your code under the MIT license.

## Credits

AirPLS baseline correction is based on the implementation by [Zhang et al.](https://pubs.rsc.org/is/content/articlelanding/2010/an/b922045c). The current implementation is based on the Python implementation by [zmzhang](https://github.com/zmzhang/airPLS).