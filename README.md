![chemotools](assets/images/banner_dark.png)

# chemotools


[![PyPI](https://img.shields.io/pypi/v/chemotools)](https://pypi.org/project/chemotools)
[![Python Versions](https://img.shields.io/pypi/pyversions/chemotools)](https://pypi.org/project/chemotools)
[![License](https://img.shields.io/pypi/l/chemotools)](https://github.com/paucablop/chemotools/blob/main/LICENSE)
[![Coverage](https://codecov.io/github/paucablop/chemotools/branch/main/graph/badge.svg?token=D7JUJM89LN)](https://codecov.io/github/paucablop/chemotools)
[![Downloads](https://static.pepy.tech/badge/chemotools)](https://pepy.tech/project/chemotools)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06802/status.svg)](https://doi.org/10.21105/joss.06802)
[![CodeFactor](https://www.codefactor.io/repository/github/paucablop/chemotools/badge/main)](https://www.codefactor.io/repository/github/paucablop/chemotools/overview/main)

---

`chemotools` is a Python library that brings **chemometric preprocessing tools** into the [`scikit-learn`](https://scikit-learn.org/) ecosystem.  

It provides modular transformers for spectral data, designed to plug seamlessly into your ML workflows.

## Features

- Preprocessing for spectral data (baseline correction, smoothing, scaling, derivatization, scatter correction).
- Physical unit conversions (absorbance, transmittance, reflectance, Kubelka-Munk, pseudoabsorbance).
- Adaptation methods for calibration transfer between instruments (DS, PDS, x-axis interpolation).
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
You can get started quickly by using the predefined [Taskfile](./Taskfile.yml), which provides handy shortcuts for common tasks:

### Setup & Validation

```bash
task install     # install all dependencies
task check       # run formatting, linting, typing, and tests
task test        # quick test run in the current environment
task coverage    # run tests with coverage reporting
```

### Testing

Run tests in your current environment:
```bash
task test           # quick test
task test:quick     # same as task test
```

Run tests across the compatibility matrix using `nox`:
```bash
task test:nox:list              # list available nox sessions
task test:nox:core              # core tests (no plotting/inspector)
task test:nox:full              # full Python 3.10-3.14 matrix
task test:nox:min-sklearn       # minimum scikit-learn compatibility tests
task test:nox:all               # run all test matrices
```

### Benchmarking

Profile performance of estimators:
```bash
task benchmark:list             # list available benchmarks
task benchmark:list:all         # show detailed benchmark registry
task benchmark:run -- --estimator adaptation.direct_standardization
task benchmark:run -- --estimator baseline.air_pls --profile regular
```

### Building & Documentation

```bash
task build              # build the package
task docs:html          # build English documentation
task docs:html-all      # build all language variants
```

For more control, use [`nox`](https://nox.thea.codes/) directly:

```bash
uv run nox --list                       # show all available sessions
uv run nox -s tests-3.12               # run tests on a specific Python version
uv run nox -s tests-min-sklearn-3.12   # test minimum scikit-learn version
```

## Contributing

Contributions are welcome!
Check out the [contributing guide](CONTRIBUTING.md) and the [project board](https://github.com/users/paucablop/projects/4).

## License

Released under the [MIT License](LICENSE).

## Compliance and Software Supply Chain Management

This project embraces software supply chain transparency by generating an SBOM (Software Bill of Materials) for all dependencies. SBOMs help organizations, including those in regulated industries, track open-source components, ensure compliance, and manage security risks. 

The SBOM file is made public as an asset attached to every release. It is generated using [CycloneDX SBOM generator for Python](https://github.com/CycloneDX/cyclonedx-python), and can be vsualized in tools like [CycloneDX Sunshine](https://cyclonedx.github.io/Sunshine/).

