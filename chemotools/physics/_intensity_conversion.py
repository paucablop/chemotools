"""
The :mod:`chemotools.physics._intensity_conversion` module implements
an IntensityConversion transformer for spectral unit conversion.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from chemotools._doc_mixin import DocLinkMixin

_VALID_UNITS = frozenset(
    {"absorbance", "transmittance", "kubelka_munk", "reflectance", "pseudoabsorbance"}
)

_VALID_CONVERSIONS = frozenset(
    {
        ("absorbance", "transmittance"),
        ("transmittance", "absorbance"),
        ("reflectance", "kubelka_munk"),
        ("kubelka_munk", "reflectance"),
        ("reflectance", "pseudoabsorbance"),
        ("pseudoabsorbance", "reflectance"),
    }
)


def _absorbance_to_transmittance(X: np.ndarray) -> np.ndarray:
    return np.power(10.0, -X)


def _transmittance_to_absorbance(X: np.ndarray) -> np.ndarray:
    near_zero = X <= 0
    if near_zero.any():
        warnings.warn(
            f"Transmittance values <= 0 found at indices "
            f"{np.argwhere(near_zero).tolist()}. "
            "These will produce inf or nan in the output.",
            UserWarning,
            stacklevel=3,
        )
    return -np.log10(X)


def _reflectance_to_kubelka_munk(X: np.ndarray) -> np.ndarray:
    near_zero = X <= 0
    if near_zero.any():
        warnings.warn(
            f"Reflectance values <= 0 found at indices "
            f"{np.argwhere(near_zero).tolist()}. "
            "These will produce inf or nan in the output.",
            UserWarning,
            stacklevel=3,
        )
    return (1.0 - X) ** 2 / (2.0 * X)


def _kubelka_munk_to_reflectance(X: np.ndarray) -> np.ndarray:
    # Inverse of F(R) = (1-R)^2 / (2R): R = (1+F) - sqrt((1+F)^2 - 1)
    negative = X < 0
    if negative.any():
        warnings.warn(
            f"Kubelka-Munk values < 0 found at indices "
            f"{np.argwhere(negative).tolist()}. "
            "These are non-physical and will produce nan in the output.",
            UserWarning,
            stacklevel=3,
        )

    return (1.0 + X) - np.sqrt((1.0 + X) ** 2 - 1.0)


def _reflectance_to_pseudoabsorbance(X: np.ndarray) -> np.ndarray:
    near_zero = X <= 0
    if near_zero.any():
        warnings.warn(
            f"Reflectance values <= 0 found at indices "
            f"{np.argwhere(near_zero).tolist()}. "
            "These will produce inf or nan in the output.",
            UserWarning,
            stacklevel=3,
        )
    return -np.log10(X)


def _pseudoabsorbance_to_reflectance(X: np.ndarray) -> np.ndarray:
    return np.power(10.0, -X)


_CONVERSION_DISPATCH = {
    ("absorbance", "transmittance"): _absorbance_to_transmittance,
    ("transmittance", "absorbance"): _transmittance_to_absorbance,
    ("reflectance", "kubelka_munk"): _reflectance_to_kubelka_munk,
    ("kubelka_munk", "reflectance"): _kubelka_munk_to_reflectance,
    ("reflectance", "pseudoabsorbance"): _reflectance_to_pseudoabsorbance,
    ("pseudoabsorbance", "reflectance"): _pseudoabsorbance_to_reflectance,
}


class IntensityConversion(
    DocLinkMixin, TransformerMixin, OneToOneFeatureMixin, BaseEstimator
):
    """
    A transformer that converts spectral intensity data between common
    measurement units used in vibrational and diffuse reflectance spectroscopy.
    For the full list of supported conversions and their equations, see the
    **Notes** section below.

    All ratio units (transmittance, reflectance) use the standard fraction
    convention (0–1). Data in percent (0–100) must be divided by 100 before
    using this transformer.

    Parameters
    ----------
    input_unit : str, default="absorbance"
        The unit of the input data. One of:
        ``"absorbance"``, ``"transmittance"``, ``"reflectance"``,
        ``"kubelka_munk"``, ``"pseudoabsorbance"``.

    output_unit : str, default="transmittance"
        The unit of the output data. One of:
        ``"absorbance"``, ``"transmittance"``, ``"reflectance"``,
        ``"kubelka_munk"``, ``"pseudoabsorbance"``.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    Examples
    --------
    >>> import numpy as np
    >>> from chemotools.physics import IntensityConversion
    >>> X = np.array([[0.0, 1.0, 2.0]])
    >>> converter = IntensityConversion(
            input_unit="absorbance", output_unit="transmittance"
        )
    IntensityConversion()
    >>> converter.fit_transform(X)
    array([[1.  , 0.1 , 0.01]])

    Notes
    -----
    Supported conversion pairs:

    - ``"absorbance"`` ↔ ``"transmittance"``:

      .. math::

         T = 10^{-A}, \\quad A = -\\log_{10}(T)

    - ``"reflectance"`` → ``"kubelka_munk"``:

      .. math::

         F(R) = \\frac{(1-R)^2}{2R}

    - ``"kubelka_munk"`` → ``"reflectance"``:

      .. math::

         R = (1+F) - \\sqrt{(1+F)^2 - 1}

    - ``"reflectance"`` ↔ ``"pseudoabsorbance"``:

      .. math::

         PA = -\\log_{10}(R), \\quad R = 10^{-PA}
    """

    _parameter_constraints: dict = {
        "input_unit": [StrOptions(_VALID_UNITS)],
        "output_unit": [StrOptions(_VALID_UNITS)],
    }

    def __init__(
        self,
        input_unit: str = "absorbance",
        output_unit: str = "transmittance",
    ):
        self.input_unit = input_unit
        self.output_unit = output_unit

    def fit(self, X: np.ndarray, y=None) -> "IntensityConversion":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to fit the transformer to.

        y : None
            Ignored to align with API.

        Returns
        -------
        self : IntensityConversion
            The fitted transformer.
        """
        self._validate_params()

        if (self.input_unit, self.output_unit) not in _VALID_CONVERSIONS:
            raise ValueError(
                f"Conversion from '{self.input_unit}' to '{self.output_unit}' is not "
                f"supported. Supported conversions are: {sorted(_VALID_CONVERSIONS)}."
            )

        validate_data(
            self, X, y="no_validation", ensure_2d=True, reset=True, dtype=np.float64
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Convert the input data to the target unit.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data to transform.

        y : None
            Ignored to align with API.

        Returns
        -------
        X_converted : np.ndarray of shape (n_samples, n_features)
            The converted data.
        """
        check_is_fitted(self, "n_features_in_")

        X_ = validate_data(
            self,
            X,
            y="no_validation",
            ensure_2d=True,
            copy=True,
            reset=False,
            dtype=np.float64,
        )

        conversion_fn = _CONVERSION_DISPATCH[(self.input_unit, self.output_unit)]
        return conversion_fn(X_)
