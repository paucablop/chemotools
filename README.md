![chemotools](assets/images/logo_pixel.png)


[![pypi](https://img.shields.io/pypi/v/chemotools)](https://pypi.org/project/chemotools)
[![pypi](https://img.shields.io/pypi/pyversions/chemotools)](https://pypi.org/project/chemotools)
[![pypi](https://img.shields.io/pypi/l/chemotools)](https://github.com/paucablop/chemotools/blob/main/LICENSE)
[![codecov](https://codecov.io/github/paucablop/chemotools/branch/main/graph/badge.svg?token=D7JUJM89LN)](https://codecov.io/github/paucablop/chemotools)
[![Downloads](https://static.pepy.tech/badge/chemotools)](https://pepy.tech/project/chemotools)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06802/status.svg)](https://doi.org/10.21105/joss.06802)


# __chemotools__

Welcome to Chemotools, a Python package that integrates chemometrics with Scikit-learn.

## Note

Since I released Chemotools, I have received a fantastic response from the community. I am really happy for the interest in the project 🤗. This also means that I have received a lot of good feedback and suggestions for improvements. I have been intensively working on releasing new versions of Chemotools to address the feedback and suggestions. If you use Chemotools, __make sure you are using the latest version__ (see installation), which will be aligned with the documentation.

👉👉 Check the [latest version](https://pypi.org/project/chemotools/) and make sure you don't miss out on cool new features.

👉👉 Check the [documentation](https://paucablop.github.io/chemotools/) for a full description on how to use chemotools.

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

Upgrading to the latest version is as simple as:

```bash
pip install chemotools --upgrade
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

## Development

To install/update the package and its development dependencies, the following command can be used:

```bash
python -m pip install --upgrade . -r requirements.txt -r requirements-dev.txt
```

``chemotools`` also comes with a ``Makefile`` that provides shortcuts for common development tasks. The equivalent command to the one above would be:

```bash
make install-dev
```

Other useful commands include:

- building the package:
    ```bash
    python -m build

    # or using the Makefile
    make build
    ```

- checking the linting of the package:
    ```bash
    flake8 ./chemotools ./tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ./chemotools ./tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    # or using the Makefile
    make lint-flake8
    ```

- testing only selected tests:
    ```bash
    pytest ./tests -k "test_load_coffee_pandas"

    # or using the Makefile
    make test TEST=test_load_coffee_pandas
    ```

- parallelized testing the package with a coverage report:
    ```bash
    pytest --cov=chemotools ./tests -n="auto" --cov-report=html -x  # for an HTML report
    pytest --cov=chemotools ./tests -n="auto" --cov-report=xml -x  # for an XML report

    # or using the Makefile
    make test-htmlcov
    make test-xmlcov
    ```

## Contributing

We welcome contributions to Chemotools from anyone interested in improving the package. Whether you have ideas for new features, bug reports, or just want to help improve the code, we appreciate your contributions! You are also welcome to see the [Project Board](https://github.com/users/paucablop/projects/4) to see what we are currently working on.

To contribute to Chemotools, please follow the [contributing guidelines](CONTRIBUTING.md).

## License

This package is distributed under the MIT license. See the [LICENSE](LICENSE) file for more information.

## Credits

AirPLS baseline correction is based on the implementation by [Zhang et al.](https://pubs.rsc.org/is/content/articlelanding/2010/an/b922045c). The current implementation is based on the Python implementation by [zmzhang](https://github.com/zmzhang/airPLS).