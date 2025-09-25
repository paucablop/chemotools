"""
Small focused tests for the Modified Sinc Smoother algorithm.

This test file contains targeted tests for specific functionality of the Modified Sinc Filter
implementation, focusing on numerical accuracy, edge cases, and specific algorithmic behavior.
"""

import numpy as np
import pytest
from chemotools.smooth import ModifiedSincFilter


class TestModifiedSincFilterSmall:
    """Small, focused tests for Modified Sinc Filter."""

    def test_impulse_response_case1(self):
        """Test specific impulse response case from reference implementation."""
        # Input: impulse at center with window_size=9, n=6, alpha=3.5
        input_signal = np.array([[0., 0., 0., 0., 1., 0., 0., 0., 0.]])
        expected_output = np.array([0.00000000, 0.00201064, -0.03742112, 0.15938869, 0.75204357, 0.15938869, -0.03742112, 0.00201064, 0.00000000])
        
        smoother = ModifiedSincFilter(window_size=9, n=6, alpha=3.5, flatten_passband=False)
        result = smoother.fit_transform(input_signal)
        
        # Test that the result is close to expected (allowing for implementation differences)
        np.testing.assert_allclose(result[0], expected_output, rtol=1e-4, atol=1e-6)

    def test_constant_signal_preservation_case2(self):
        """Test constant signal preservation with window_size=5, n=6, alpha=3.0."""
        # Input: constant signal should remain unchanged
        input_signal = np.array([[3., 3., 3., 3., 3., 3.]])
        expected_output = np.array([3., 3., 3., 3., 3., 3.])
        
        smoother = ModifiedSincFilter(window_size=5, n=6, alpha=3.0, flatten_passband=False)
        result = smoother.fit_transform(input_signal)
        
        # Constant signals should be preserved exactly (DC preservation)
        np.testing.assert_allclose(result[0], expected_output, rtol=1e-10, atol=1e-12)

    def test_single_point_signal(self):
        """Test behavior with single data point."""
        input_signal = np.array([[5.0]])
        
        smoother = ModifiedSincFilter(window_size=3, n=2, alpha=2.0)
        result = smoother.fit_transform(input_signal)
        
        # Single point should remain unchanged
        assert result.shape == input_signal.shape
        np.testing.assert_allclose(result, input_signal, rtol=1e-12)

    def test_two_point_signal(self):
        """Test behavior with two data points."""
        input_signal = np.array([[1.0, 2.0]])
        
        smoother = ModifiedSincFilter(window_size=3, n=2, alpha=2.0)
        result = smoother.fit_transform(input_signal)
        
        # Should handle gracefully
        assert result.shape == input_signal.shape
        assert np.all(np.isfinite(result))

    def test_kernel_properties(self):
        """Test mathematical properties of the generated kernel."""
        smoother = ModifiedSincFilter(window_size=11, n=4, alpha=2.5)
        smoother.fit(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]))
        
        kernel = smoother.kernel_
        
        # Test kernel properties
        assert len(kernel) == 11
        assert np.isclose(np.sum(kernel), 1.0, rtol=1e-10)  # DC preservation
        
        # Test symmetry
        np.testing.assert_allclose(kernel, kernel[::-1], rtol=1e-12)
        
        # Center should be the maximum for a low-pass filter
        center_idx = len(kernel) // 2
        assert kernel[center_idx] == np.max(kernel)

    def test_different_window_sizes(self):
        """Test with various odd window sizes."""
        input_signal = np.array([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
        
        for window_size in [3, 5, 7, 9]:
            smoother = ModifiedSincFilter(window_size=window_size, n=4, alpha=2.0)
            result = smoother.fit_transform(input_signal)
            
            assert result.shape == input_signal.shape
            assert np.all(np.isfinite(result))
            
            # Smoothed signal should be less "rough" than original
            # (smaller total variation for typical signals)
            original_variation = np.sum(np.abs(np.diff(input_signal[0])))
            smoothed_variation = np.sum(np.abs(np.diff(result[0])))
            assert smoothed_variation <= original_variation

    def test_alpha_parameter_effect(self):
        """Test effect of alpha parameter on smoothing strength."""
        # Step function - good test signal for smoothing
        input_signal = np.array([[0., 0., 0., 1., 1., 1., 0., 0., 0.]])
        
        results = {}
        for alpha in [1.0, 2.0, 4.0]:
            smoother = ModifiedSincFilter(window_size=7, n=4, alpha=alpha)
            results[alpha] = smoother.fit_transform(input_signal)[0]
        
        # Higher alpha should produce more smoothing (lower peak at step)
        step_idx = 4  # Middle of the step
        assert results[4.0][step_idx] < results[2.0][step_idx] < results[1.0][step_idx]

    def test_n_parameter_effect(self):
        """Test effect of n parameter on filter characteristics."""
        input_signal = np.array([[0., 0., 0., 1., 0., 0., 0.]])
        
        results = {}
        for n in [2, 4, 6]:
            smoother = ModifiedSincFilter(window_size=7, n=n, alpha=2.0)
            results[n] = smoother.fit_transform(input_signal)[0]
        
        # All should preserve the general shape but with different characteristics
        for n in [2, 4, 6]:
            assert np.argmax(results[n]) == 3  # Peak should remain at center
            assert np.isclose(np.sum(results[n]), 1.0, rtol=1e-10)  # DC preservation

    def test_flatten_passband_effect(self):
        """Test the effect of passband flattening."""
        # Sinusoidal signal that should be preserved better with flattening
        x = np.linspace(0, 2*np.pi, 20)
        input_signal = np.sin(x).reshape(1, -1)
        
        smoother_flat = ModifiedSincFilter(window_size=7, n=4, alpha=2.0, flatten_passband=True)
        smoother_no_flat = ModifiedSincFilter(window_size=7, n=4, alpha=2.0, flatten_passband=False)
        
        result_flat = smoother_flat.fit_transform(input_signal)
        result_no_flat = smoother_no_flat.fit_transform(input_signal)
        
        # Both should be finite and preserve general shape
        assert np.all(np.isfinite(result_flat))
        assert np.all(np.isfinite(result_no_flat))
        
        # Results should be different when flattening is applied
        assert not np.allclose(result_flat, result_no_flat, rtol=1e-6)

    def test_boundary_modes(self):
        """Test different boundary handling modes."""
        input_signal = np.array([[1., 2., 3., 4., 5.]])
        
        modes = ["mirror", "constant", "nearest", "wrap", "interp"]
        results = {}
        
        for mode in modes:
            smoother = ModifiedSincFilter(window_size=5, n=4, alpha=2.0, mode=mode)
            results[mode] = smoother.fit_transform(input_signal)[0]
        
        # All modes should produce finite results
        for mode in modes:
            assert np.all(np.isfinite(results[mode]))
            assert len(results[mode]) == 5
        
        # Different modes should generally produce different results at boundaries
        # (except for special cases)
        boundary_differences = []
        modes_list = list(modes)
        for i in range(len(modes_list)):
            for j in range(i+1, len(modes_list)):
                mode1, mode2 = modes_list[i], modes_list[j]
                # Check first and last elements (boundaries)
                diff = abs(results[mode1][0] - results[mode2][0]) + abs(results[mode1][-1] - results[mode2][-1])
                boundary_differences.append(diff)
        
        # At least some boundary modes should produce different results
        assert np.max(boundary_differences) > 1e-10

    def test_noise_reduction(self):
        """Test that the filter reduces high-frequency noise."""
        # Create a signal with high-frequency noise
        np.random.seed(42)
        x = np.linspace(0, 4*np.pi, 50)
        clean_signal = np.sin(x)
        noisy_signal = clean_signal + 0.1 * np.random.randn(50)
        
        input_signal = noisy_signal.reshape(1, -1)
        
        smoother = ModifiedSincFilter(window_size=11, n=6, alpha=3.0)
        smoothed = smoother.fit_transform(input_signal)[0]
        
        # Smoothed signal should be closer to clean signal than noisy signal
        error_original = np.mean((noisy_signal - clean_signal)**2)
        error_smoothed = np.mean((smoothed - clean_signal)**2)
        
        assert error_smoothed < error_original

    def test_multiple_samples(self):
        """Test processing multiple samples simultaneously."""
        # Multiple different signals
        signals = np.array([
            [1., 2., 3., 4., 5.],
            [5., 4., 3., 2., 1.],
            [1., 1., 1., 1., 1.]
        ])
        
        smoother = ModifiedSincFilter(window_size=5, n=4, alpha=2.0)
        result = smoother.fit_transform(signals)
        
        assert result.shape == signals.shape
        assert np.all(np.isfinite(result))
        
        # Third signal (constant) should remain approximately constant
        np.testing.assert_allclose(result[2], np.ones(5), rtol=1e-6)