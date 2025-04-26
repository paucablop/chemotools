import numpy as np

from ._base import ModelTypes


def calculate_decoded_spectrum(transfomed_spectrum: np.ndarray, estimator: ModelTypes):
    """
    Calculate the decoded spectrum for a given transformed spectrum and estimator from the latent space.

    Parameters
    ----------
    transfomed_spectrum : np.ndarray
        The transformed spectrum data.

    estimator : ModelTypes
        The fitted PCA or PLS model.

    Returns
    -------
    np.ndarray
        The decoded spectrum.
    """
    # Project the transformed spectrum onto the latent space
    X_transformed = estimator.transform(transfomed_spectrum)
    
    # Decode the spectrum back to the original space
    return estimator.inverse_transform(X_transformed)


def calculate_residual_spectrum(transfomed_spectrum: np.ndarray, estimator: ModelTypes):
    """
    Calculate the residual spectrum for a given transformed spectrum and estimator.

    Parameters
    ----------
    transfomed_spectrum : np.ndarray
        The transformed spectrum data.

    estimator : ModelTypes
        The fitted PCA or PLS model.

    Returns
    -------
    np.ndarray
        The residual spectrum.
    """
    # Compute the reconstruction error (Q residuals)
    decoded_spectrum = calculate_decoded_spectrum(transfomed_spectrum, estimator)

    # Calculate the residual
    return transfomed_spectrum - decoded_spectrum


