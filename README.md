![chemotools](assets/images/banner_dark.png)

# chemotools


[![PyPI](https://img.shields.io/pypi/v/chemotools)](https://pypi.org/project/chemotools)
[![Python Versions](https://img.shields.io/pypi/pyversions/chemotools)](https://pypi.org/project/chemotools)
[![License](https://img.shields.io/pypi/l/chemotools)](https://github.com/paucablop/chemotools/blob/main/LICENSE)
[![Coverage](https://codecov.io/github/paucablop/chemotools/branch/main/graph/badge.svg?token=D7JUJM89LN)](https://codecov.io/github/paucablop/chemotools)
[![Downloads](https://static.pepy.tech/badge/chemotools)](https://pepy.tech/project/chemotools)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06802/status.svg)](https://doi.org/10.21105/joss.06802)

---

`chemotools` is a Python library that brings **chemometric preprocessing tools** into the [`scikit-learn`](https://scikit-learn.org/) ecosystem.  

It provides modular transformers for spectral data, designed to plug seamlessly into your ML workflows.

## Features

- Preprocessing for spectral data (baseline correction, smoothing, scaling, derivatization, scatter correction).  
- Fully compatible with `scikit-learn` pipelines and transformers.  
- Simple, modular API for flexible workflows.  
- Open-source, actively maintained, and published on [PyPI](https://pypi.org/project/chemotools/) and [Conda](https://anaconda.org/conda-forge/chemotools).  

## Installation

Install from PyPI:

```bash
pip install chemotools
````

Install from Conda:

```bash
conda install -c conda-forge chemotools
```

## Usage

Example: preprocessing pipeline with scikit-learn:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from chemotools.baseline import AirPls
from chemotools.scatter import MultiplicativeScatterCorrection

preprocessing = make_pipeline(
    AirPls(),
    MultiplicativeScatterCorrection(),
    StandardScaler(with_std=False),
)

spectra_transformed = preprocessing.fit_transform(spectra)
```

➡️ See the [documentation](https://paucablop.github.io/chemotools/) for full details.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [Task](https://taskfile.dev) to simplify common development workflows.
You can get started quickly by using the predefined [Taskfile](./Taskfile.yml), which provides handy shortcuts such as:

```bash
task install     # install all dependencies
task check       # run formatting, linting, typing, and tests
task coverage    # run tests with coverage reporting
task build       # build the package for distribution
```

## Contributing

Contributions are welcome!
Check out the [contributing guide](CONTRIBUTING.md) and the [project board](https://github.com/users/paucablop/projects/4).

## License

Released under the [MIT License](LICENSE).

