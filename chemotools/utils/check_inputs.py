from sklearn.utils.validation import check_array


def check_input(X, y=None):
    # Check that X is a 2D array and has only finite values
    X = check_array(X, ensure_2d=True, force_all_finite=True)

    # Check that y is None or a 1D array of the same length as X
    if y is not None:
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        y = check_array(y, force_all_finite=True)
        if len(y) != X.shape[0]:
            raise ValueError("y must have the same number of samples as X")
    return X
