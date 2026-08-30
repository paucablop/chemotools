# Transformer Benchmark Checklist

Use this checklist to track benchmark completion per transformer.

Legend for `Done`: `- [ ]` not started, `- [x]` completed.

Columns:
- Baseline mean (ms/call): mean runtime before optimization
- Candidate mean (ms/call): mean runtime after optimization
- Speed gain (%): `((baseline - candidate) / baseline) * 100`
- Speedup (x): `baseline / candidate`

## adaptation

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | DirectStandardization | chemotools/adaptation/_direct_standardization.py | 377.722 | 27.674 | -92.67% | 13.65x | Store SVD of `T_` instead of full `T_` because often `T_` is not full rank (`n_samples` << `n_features`)|
| - [x] | PiecewiseDirectStandardization | chemotools/adaptation/_piecewise_direct_standardization.py | 395.252271 | 239.677417 | -39.19% | 1.64x | Change sparse representation for banded |
| - [x] | SpectralSpaceTransform | chemotools/adaptation/_spectral_space_transform.py | 10.235208 | NA | NA | NA | Already optimized |
| - [x] | XAxisInterpolator | chemotools/adaptation/_x_axis_interpolator.py | 688.784 | 381.067 | -44.68% | 1.81x | Parallelization |

## augmentation

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [ ] | AddNoise | chemotools/augmentation/_add_noise.py |  |  |  |  |  |
| - [ ] | BaselineShift | chemotools/augmentation/_baseline_shift.py |  |  |  |  |  |
| - [ ] | FractionalShift | chemotools/augmentation/_fractional_shift.py |  |  |  |  |  |
| - [ ] | GaussianBroadening | chemotools/augmentation/_gaussian_broadening.py |  |  |  |  |  |
| - [ ] | IndexShift | chemotools/augmentation/_index_shift.py |  |  |  |  |  |
| - [ ] | SpectrumScale | chemotools/augmentation/_spectrum_scale.py |  |  |  |  |  |

## baseline

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | AirPls | chemotools/baseline/_air_pls.py | 10518.504 | 2440.694 | -76.80% | 4.31x | Parallelize code + warm-start |
| - [x] | ArPls | chemotools/baseline/_ar_pls.py | 4168.078 | 1113.196 | -73.30% | 3.74x | Parallelize code + warm-start |
| - [x] | AsLs | chemotools/baseline/_as_ls.py | 6229.270 | 1532.880 | -75.40% | 4.06x | Parallelize code + warm-start |
| - [x] | ConstantBaselineCorrection | chemotools/baseline/_constant_baseline_correction.py | 37.408 | 17.355 | -53.61% | 2.16x | Vectorize |
| - [x] | CubicSplineCorrection | chemotools/baseline/_cubic_spline_correction.py | 1226.286 | 131.517 | -89.28% | 9.32x | Vectorize |
| - [x] | LinearCorrection | chemotools/baseline/_linear_correction.py | 77.849 | 29.994 | -61.47% |  2.60x | Vectorization |
| - [x] | NonNegative | chemotools/baseline/_non_negative.py | 34.895 | 16.959 | -51.40% | 2.06x | Vectorization |
| - [x] | PolynomialCorrection | chemotools/baseline/_polynomial_correction.py | 211.392 | 0.314 | -99.85% | 673.04x | Vectorization (benchmark done on reduced dataset, 100 samples / 1000 features / 1 run) |
| - [x] | RubberbandCorrection | chemotools/baseline/_rubberband_correction.py | 7181.486 | 634.416 | -91.17% | 11.32x | Parallelize + use scipy ConvexHull |
| - [x] | SubtractReference | chemotools/baseline/_subtract_reference.py | 18.1044165 | NA | NA | NA | Already optimized |

## derivative

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | NorrisWilliams | chemotools/derivative/_norris_william.py | 152.926 | 58.443 | -61.78% | 2.62x | Extreme values changed due to fused kernel, vectorization, fused kernel calculated during fit|
| - [x] | SavitzkyGolay | chemotools/derivative/_savitzky_golay.py | 185.028 | 44.221 | -76.10% | 4.18x | Vectorize |

## physics

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | IntensityConversion | chemotools/physics/_intensity_conversion.py |  167.76022079999998 | NA | NA | NA | Alrady Vectorized |

## projection

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | DirectOrthogonalization | chemotools/projection/_direct_orthogonalization.py | 41.929682979999995 | NA | NA | NA | NA |
| - [x] | ExternalParameterOrthogonalization | chemotools/projection/_external_parameter_orthogonalization.py | 397.053 | 38.532 | -90.30% | 10.30x |Store Key SVD from `P_epo` and not full `P_epo` |
| - [x] | OrthogonalPLS | chemotools/projection/_orthogonal_pls.py | 36.54471162 | NA | NA | NA | NA |
| - [x] | OrthogonalSignalCorrection | chemotools/projection/_orthogonal_signal_correction.py |  58.1801665 (wold), 58.123208000000005 (fearn), 59.3155205 (soblom) | NA | NA | NA | NA |

## regression

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | PLSRegression | chemotools/regression/_pls_regression.py | 29.563048270000003 | NA | NA | NA | NA |

## scale

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | BandScaler | chemotools/scale/_band_scaler.py | 21.149896 | NA | NA | NA | Already optimized |
| - [x] | MinMaxScaler | chemotools/scale/_min_max_scaler.py | 35.501 | 31.899 | -10.15% | 1.11x | Avoid double calculation of np.min |
| - [x] | NormScaler | chemotools/scale/_norm_scaler.py | 42.153 | 27.534 | -34.68% | 1.53x | Vectorize |
| - [x] | ParetoScaler | chemotools/scale/_pareto_scaler.py | 26.9398125 | NA | NA | NA | Already optimize |
| - [x] | PointScaler | chemotools/scale/_point_scaler.py | 19.979 | 17.773 | -11.04% | 1.12x | Vectorize |

## scatter

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | ExtendedMultiplicativeScatterCorrection | chemotools/scatter/_extended_multiplicative_scatter_correction.py | 203.346 | 172.065 | -15.38% | 1.18x | NA |
| - [x] | MultiplicativeScatterCorrection | chemotools/scatter/_multiplicative_scatter_correction.py | NA | NA | NA | NA | Already optimized |
| - [x] | RobustNormalVariate | chemotools/scatter/_robust_normal_variate.py | 567.398 | 90.901 | -83.98% | 6.24x | Vectorize + Parallelize|
| - [x] | StandardNormalVariate | chemotools/scatter/_standard_normal_variate.py | 101.967 | 49.584 | -51.37% | 2.06x | Vectorize |

## smooth

| Done | Transformer | File | Baseline mean (ms/call) | Candidate mean (ms/call) | Speed gain (%) | Speedup (x) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| - [x] | MeanFilter | chemotools/smooth/_mean_filter.py | 68.116 | 41.983 | -38.37% | 1.62x | Vectorize |
| - [x] | MedianFilter | chemotools/smooth/_median_filter.py | 359.997 | 81.586 | -77.34% | 4.41x | Parallelize code |
| - [x] | ModifiedSincFilter | chemotools/smooth/_modified_sinc_smoother.py | 207.8507966 | NA | NA | NA | NA |
| - [x] | SavitzkyGolayFilter | chemotools/smooth/_savitzky_golay_filter.py | 87.8475 | NA | NA | NA | NA |
| - [x] | WhittakerSmooth | chemotools/smooth/_whittaker_smooth.py | 6158.040 | 262.880 | -95.73% | 23.42x | Batched LAPACK call (solve_batch) + Parallelize code |
