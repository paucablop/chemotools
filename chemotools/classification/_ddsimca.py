from numbers import Integral

import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class DDSIMCA(BaseEstimator):
    """Data-Driven Soft Independent Modelling of Class Analogies.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components retained for each class model.

    limit_type : {"classic", "robust"}, default="classic"
        Method used to estimate the scaled chi-square parameters.

    alpha : float, default=0.05
        Significance level used to calculate the acceptance limits.
    """

    def __init__(
        self,
        n_components: int = 2,
        limit_type: str = "classic",
        alpha: float = 0.05,
    ):
        self.n_components = n_components
        self.limit_type = limit_type
        self.alpha = alpha

    @staticmethod
    def _get_distparams(
        distances: np.ndarray,
        limit_type: str = "classic",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate scaled chi-square distribution parameters.

        Parameters
        ----------
        distances : ndarray of shape (n_samples, n_distances)
            Distance values for which distribution parameters are
            estimated.

        limit_type : {"classic", "robust"}, default="classic"
            Estimation method.

        Returns
        -------
        scale : ndarray of shape (n_distances,)
            Estimated scale parameters.

        degrees_of_freedom : ndarray of shape (n_distances,)
            Estimated chi-square degrees of freedom.

        Raises
        ------
        ValueError
            If ``distances`` is not a two-dimensional array or if
            ``limit_type`` is invalid.
        """
        distances = np.asarray(distances, dtype=np.float64)

        if distances.ndim != 2:
            raise ValueError("distances must be a two-dimensional array.")

        if distances.shape[0] < 2:
            raise ValueError("At least two distance observations are required.")

        if limit_type == "classic":
            scale = distances.mean(axis=0)
            variance = distances.var(axis=0, ddof=1)

            squared_scale = np.maximum(scale**2, 1e-12)
            variance = np.maximum(variance, 1e-12)

            degrees_of_freedom = np.round(2.0 * squared_scale / variance)
            degrees_of_freedom = np.clip(
                degrees_of_freedom,
                1.0,
                250.0,
            )

            scale = np.maximum(scale, 1e-12)

            return scale, degrees_of_freedom

        if limit_type != "robust":
            raise ValueError("limit_type must be either 'classic' or 'robust'.")

        median = np.median(distances, axis=0)
        interquartile_range = np.quantile(distances, 0.75, axis=0) - np.quantile(
            distances, 0.25, axis=0
        )

        safe_median = np.maximum(np.abs(median), 1e-12)
        relative_spread = interquartile_range / safe_median

        degrees_of_freedom = np.zeros(
            relative_spread.shape,
            dtype=np.float64,
        )

        degrees_of_freedom[relative_spread > 2.685592117] = 1.0
        degrees_of_freedom[relative_spread < 0.194565995] = 100.0

        intermediate = (relative_spread >= 0.194565995) & (
            relative_spread <= 2.685592117
        )

        degrees_of_freedom[intermediate] = np.round(
            np.exp(
                (1.380948 * np.log(2.68631 / relative_spread[intermediate])) ** 1.185785
            )
        )

        degrees_of_freedom = np.clip(
            degrees_of_freedom,
            1.0,
            250.0,
        )

        median_chi2 = chi2.ppf(
            0.50,
            degrees_of_freedom,
        )
        interquartile_chi2 = chi2.ppf(0.75, degrees_of_freedom) - chi2.ppf(
            0.25, degrees_of_freedom
        )

        median_chi2 = np.maximum(median_chi2, 1e-12)
        interquartile_chi2 = np.maximum(
            interquartile_chi2,
            1e-12,
        )

        scale = (
            0.5
            * degrees_of_freedom
            * (safe_median / median_chi2 + interquartile_range / interquartile_chi2)
        )
        scale = np.maximum(scale, 1e-12)

        return scale, degrees_of_freedom

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "DDSIMCA":
        """Fit one DD-SIMCA model for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            dtype=np.float64,
            ensure_min_samples=2,
            ensure_min_features=1,
        )

        check_classification_targets(y)
        self._validate_parameters(X)

        self.classes_ = np.unique(y)

        if self.classes_.size < 2:
            raise ValueError("DDSIMCA requires at least two target classes.")

        self.mean_ = []
        self.eig_val_ = []
        self.eig_vec_ = []
        self.evr_ = []
        self.Q_train_ = []
        self.T2_train_ = []
        self.h0_ = []
        self.Nh_ = []
        self.q0_ = []
        self.Nq_ = []
        self.T2_cut_chi2_ = []
        self.Q_cut_chi2_ = []
        self.Nf_ = []
        self.F_cut_chi2_ = []

        for class_label in self.classes_:
            class_mask = y == class_label
            X_class = X[class_mask]

            if X_class.shape[0] < 2:
                raise ValueError("Each class must contain at least two samples.")

            class_mean = X_class.mean(axis=0)
            X_centered = X_class - class_mean

            covariance = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

            eigenvalues, eigenvectors = np.linalg.eigh(covariance)

            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            eigenvalues = np.maximum(eigenvalues, 0.0)

            total_variance = eigenvalues.sum()

            if total_variance <= 1e-12:
                raise ValueError(
                    "DDSIMCA cannot be fitted to a class with zero variance."
                )

            selected_eigenvalues = eigenvalues[: self.n_components]
            selected_eigenvectors = eigenvectors[
                :,
                : self.n_components,
            ]

            scores = X_centered @ selected_eigenvectors
            reconstructed = scores @ selected_eigenvectors.T

            residual_distances = np.sum(
                (X_centered - reconstructed) ** 2,
                axis=1,
            )
            score_distances = np.sum(
                scores**2 / (selected_eigenvalues + 1e-12),
                axis=1,
            )

            h0, nh = self._get_distparams(
                score_distances.reshape(-1, 1),
                self.limit_type,
            )
            q0, nq = self._get_distparams(
                residual_distances.reshape(-1, 1),
                self.limit_type,
            )

            combined_degrees_of_freedom = nh + nq

            score_cutoff = chi2.ppf(1.0 - self.alpha, nh) * h0 / nh
            residual_cutoff = chi2.ppf(1.0 - self.alpha, nq) * q0 / nq
            combined_cutoff = chi2.ppf(
                1.0 - self.alpha,
                combined_degrees_of_freedom,
            )

            self.mean_.append(class_mean)
            self.eig_val_.append(selected_eigenvalues)
            self.eig_vec_.append(selected_eigenvectors)
            self.evr_.append(selected_eigenvalues / total_variance)
            self.Q_train_.append(residual_distances)
            self.T2_train_.append(score_distances)
            self.h0_.append(h0)
            self.Nh_.append(nh)
            self.q0_.append(q0)
            self.Nq_.append(nq)
            self.T2_cut_chi2_.append(score_cutoff)
            self.Q_cut_chi2_.append(residual_cutoff)
            self.Nf_.append(combined_degrees_of_freedom)
            self.F_cut_chi2_.append(combined_cutoff)

        return self

    def _validate_parameters(
        self,
        X: np.ndarray,
    ) -> None:
        """Validate constructor parameters."""
        if (
            isinstance(self.n_components, bool)
            or not isinstance(self.n_components, Integral)
            or self.n_components < 1
        ):
            raise ValueError("n_components must be a positive integer.")

        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed the number of features.")

        if self.limit_type not in {"classic", "robust"}:
            raise ValueError("limit_type must be either 'classic' or 'robust'.")

        if (
            isinstance(self.alpha, bool)
            or not isinstance(
                self.alpha,
                (int, float, np.integer, np.floating),
            )
            or not 0.0 < self.alpha < 1.0
        ):
            raise ValueError("alpha must be a number strictly between 0 and 1.")

    def transform(
        self,
        X: np.ndarray,
    ) -> list[np.ndarray]:
        """Project samples onto each class-specific PCA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        transformed : list of ndarray
            One score matrix for each fitted class.
        """
        check_is_fitted(
            self,
            attributes=[
                "classes_",
                "mean_",
                "eig_vec_",
            ],
        )

        X = validate_data(
            self,
            X=X,
            reset=False,
            dtype=np.float64,
        )

        transformed = []

        for class_index in range(len(self.classes_)):
            X_centered = X - self.mean_[class_index]
            scores = X_centered @ self.eig_vec_[class_index]
            transformed.append(scores)

        return transformed

    def _normalized_distances(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Calculate normalized DD-SIMCA distances."""
        check_is_fitted(
            self,
            attributes=[
                "classes_",
                "mean_",
                "eig_val_",
                "eig_vec_",
                "h0_",
                "Nh_",
                "q0_",
                "Nq_",
                "F_cut_chi2_",
            ],
        )

        X = validate_data(
            self,
            X=X,
            reset=False,
            dtype=np.float64,
        )

        distances = np.empty(
            (X.shape[0], len(self.classes_)),
            dtype=np.float64,
        )

        for class_index in range(len(self.classes_)):
            X_centered = X - self.mean_[class_index]

            scores = X_centered @ self.eig_vec_[class_index]

            score_distances = np.sum(
                scores**2 / (self.eig_val_[class_index] + 1e-12),
                axis=1,
            )

            reconstructed = scores @ self.eig_vec_[class_index].T

            residual_distances = np.sum(
                (X_centered - reconstructed) ** 2,
                axis=1,
            )

            combined_distance = (score_distances / self.h0_[class_index]) * self.Nh_[
                class_index
            ] + (residual_distances / self.q0_[class_index]) * self.Nq_[class_index]

            normalized = combined_distance / self.F_cut_chi2_[class_index]

            distances[:, class_index] = np.ravel(normalized)

        return distances

    # def predict(
    #     self,
    #     X: np.ndarray,
    # ) -> np.ndarray:
    #     """Predict one class label for each sample.

    #     The class having the smallest normalized DD-SIMCA distance
    #     is selected. This method always returns one label per sample,
    #     as required by the scikit-learn classifier API.

    #     Parameters
    #     ----------
    #     X : array-like of shape (n_samples, n_features)
    #         Samples to classify.

    #     Returns
    #     -------
    #     labels : ndarray of shape (n_samples,)
    #         Predicted class labels.
    #     """
    #     distances = self._normalized_distances(X)
    #     closest_class_indices = np.argmin(
    #         distances,
    #         axis=1,
    #     )

    #     return self.classes_[closest_class_indices]

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return the DD-SIMCA acceptance decision for each class.

        Unlike ``predict``, this method preserves the soft-classification
        behaviour of DD-SIMCA. A sample may be accepted by zero, one, or
        multiple classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        acceptance : ndarray of shape (n_samples, n_classes)
            Boolean acceptance matrix. Column order corresponds to
            ``classes_``.
        """
        distances = self._normalized_distances(X)

        return distances <= 1.0
