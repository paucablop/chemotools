# Inspector Module Tests

## Overview
This document describes the test suite created for the `chemotools.inspector` module.

## Test Files Created

### 1. `tests/inspector/test_validate.py`
Tests for the validation functions in `chemotools/inspector/_validate.py`.

#### TestValidateAndExtractModel
- **test_validate_fitted_pca**: Validates extraction of a fitted PCA model
- **test_validate_fitted_pls**: Validates extraction of a fitted PLS model
- **test_validate_fitted_pipeline_pca**: Validates extraction from a pipeline with preprocessing
- **test_unfitted_pca_raises_error**: Ensures unfitted PCA raises NotFittedError
- **test_unfitted_pls_raises_error**: Ensures unfitted PLS raises NotFittedError
- **test_unfitted_pipeline_raises_error**: Ensures unfitted pipeline raises NotFittedError
- **test_invalid_model_type_raises_error**: Ensures invalid model types raise TypeError
- **test_pipeline_with_invalid_final_step**: Ensures pipelines ending with non-PCA/PLS raise TypeError
- **test_single_step_pipeline**: Tests pipeline with only PCA (no preprocessing)
- **test_multi_step_pipeline**: Tests pipeline with multiple preprocessing steps

#### TestValidateDatasetsConsistency
- **test_valid_train_only_unsupervised**: Validates training data only for unsupervised learning
- **test_valid_train_test_unsupervised**: Validates train/test split for unsupervised learning
- **test_valid_all_datasets_unsupervised**: Validates train/test/val split for unsupervised learning
- **test_valid_train_only_supervised**: Validates training data only for supervised learning
- **test_valid_train_test_supervised**: Validates train/test split for supervised learning
- **test_inconsistent_features_in_test**: Ensures test set with wrong features raises ValueError
- **test_inconsistent_features_in_val**: Ensures validation set with wrong features raises ValueError
- **test_supervised_missing_y_train**: Ensures supervised learning without y_train raises ValueError
- **test_supervised_missing_y_test**: Ensures supervised learning with X_test but no y_test raises ValueError
- **test_supervised_missing_y_val**: Ensures supervised learning with X_val but no y_val raises ValueError

### 2. `tests/inspector/test_base.py`
Tests for the base inspector class in `chemotools/inspector/_base.py`.

#### TestBaseInspectorInitialization
- **test_init_with_fitted_pca**: Tests initialization with a fitted PCA model
- **test_init_with_fitted_pls**: Tests initialization with a fitted PLS model
- **test_init_with_pipeline**: Tests initialization with a fitted pipeline
- **test_init_with_test_data**: Tests initialization including test data
- **test_init_with_validation_data**: Tests initialization including validation data
- **test_init_with_all_datasets**: Tests initialization with train/test/val splits
- **test_init_with_feature_names**: Tests initialization with feature names
- **test_init_with_sample_labels**: Tests initialization with sample labels
- **test_init_with_unfitted_model**: Ensures unfitted model raises error
- **test_init_with_invalid_model**: Ensures invalid model type raises error

#### TestBaseInspectorOrganizeDatasets
- **test_organize_train_only**: Tests organizing training data only
- **test_organize_train_and_test**: Tests organizing train and test data
- **test_organize_all_datasets**: Tests organizing all dataset splits

#### TestBaseInspectorGetNComponents
- **test_get_n_components_from_pca**: Tests extracting n_components from PCA
- **test_get_n_components_from_pls**: Tests extracting n_components from PLS
- **test_get_n_components_different_values**: Tests with different n_components values

#### TestBaseInspectorTransformData
- **test_transform_without_pipeline**: Tests transformation without preprocessing
- **test_transform_with_pipeline**: Tests transformation with preprocessing pipeline

#### TestBaseInspectorGetScores
- **test_get_scores_pca**: Tests getting scores from PCA model
- **test_get_scores_pls**: Tests getting scores from PLS model
- **test_get_scores_with_pipeline**: Tests getting scores with preprocessing pipeline
- **test_get_scores_different_dataset**: Tests getting scores for different dataset splits

#### TestBaseInspectorPlotScores
- **test_plot_scores_is_abstract**: Verifies plot_scores is implemented
- **test_plot_scores_basic**: Tests basic plotting functionality
- **test_plot_scores_multiple_datasets**: Tests plotting multiple datasets
- **test_plot_scores_different_components**: Tests plotting different component pairs

#### TestBaseInspectorProperties
- **test_n_features_in**: Tests n_features_in_ attribute
- **test_datasets_structure**: Tests datasets_ structure
- **test_estimator_attribute**: Tests estimator_ attribute
- **test_transformer_attribute_none**: Tests transformer_ when None
- **test_transformer_attribute_pipeline**: Tests transformer_ with pipeline

## Test Coverage Summary

The test suite provides comprehensive coverage of:
- ✅ Model validation (fitted/unfitted, valid/invalid types)
- ✅ Pipeline handling (single-step, multi-step, extraction of estimator and transformer)
- ✅ Dataset organization (train, test, validation splits)
- ✅ Dataset consistency validation (feature counts, supervised vs unsupervised)
- ✅ Component extraction from different model types
- ✅ Data transformation (with and without preprocessing)
- ✅ Score calculation for different models and datasets
- ✅ Plotting functionality
- ✅ Attribute initialization and access
- ✅ Error handling for edge cases

## Additional Fixture Added

Added `fitted_invalid_model` fixture to `tests/conftest.py` for testing invalid model types that are fitted (to distinguish type errors from fitting errors).

## Running the Tests

```bash
# Run all inspector tests
pytest tests/inspector/

# Run specific test file
pytest tests/inspector/test_validate.py
pytest tests/inspector/test_base.py

# Run with verbose output
pytest tests/inspector/ -v

# Run with coverage
pytest tests/inspector/ --cov=chemotools.inspector
```

## Notes

- All tests follow the Arrange-Act-Assert pattern
- Tests use existing fixtures from `conftest.py`
- Mock ConcreteInspector class is used to test the abstract _BaseInspector
- Tests verify both positive cases and error conditions
- Matplotlib figures are properly closed to avoid memory leaks
