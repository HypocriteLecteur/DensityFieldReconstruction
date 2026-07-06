import numpy as np
import pytest

from dfr.analysis import (
    ManifoldAnalysisResult,
    centered_3pl_excess,
    fit_centered_3pl_curves,
    fit_symmetric_2pl_curves,
    load_legacy_manifold_cache,
    project_to_shape_curve,
    scale_for_mode_count,
    symmetric_2pl_mode_count,
)


JACKDAW_CACHE_ROW = np.array(
    [
        66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 40, 36, 32, 28, 27,
        25, 22, 21, 21, 21, 19, 18, 15, 15, 12, 11, 10, 10, 9, 6, 5, 5,
        5, 4, 4, 4, 3, 3, 2,
    ],
    dtype=float,
)
JACKDAW_SCALE_RANGE = np.array([0.36664170535718477, 5.432115162711986])


def test_legacy_cache_loader_uses_historic_three_file_schema(tmp_path):
    np.save(tmp_path / "modes.npy", JACKDAW_CACHE_ROW[None, :])
    np.save(tmp_path / "scale_range.npy", JACKDAW_SCALE_RANGE[None, :])
    np.save(tmp_path / "nn_dists.npy", np.array([0.72]))

    cache = load_legacy_manifold_cache(tmp_path)

    np.testing.assert_array_equal(cache.mode_counts[0], JACKDAW_CACHE_ROW)
    np.testing.assert_allclose(cache.scale_ranges[0], JACKDAW_SCALE_RANGE)
    np.testing.assert_array_equal(cache.completed_rows, [True])


def test_extracted_centered_3pl_fit_matches_historic_cache_golden_values():
    fitted = fit_centered_3pl_curves(
        [350],
        [66],
        JACKDAW_SCALE_RANGE[None, :],
        JACKDAW_CACHE_ROW[None, :],
        dataset_name="jackdaw",
    )

    assert fitted.success.tolist() == [True]
    assert fitted.result.parameter_names == ("k", "sigma_half", "log10_gamma")
    np.testing.assert_allclose(
        fitted.result.parameters[0],
        [1.548541477753584, 0.8844357111994945, 0.12979024999535804],
        rtol=1e-7,
    )
    np.testing.assert_allclose(fitted.residual_variances, [2.574235909279746])
    assert fitted.result.dataset_names.tolist() == ["jackdaw"]


def test_recommended_scale_inverts_centered_3pl():
    parameters = np.array([2.4, 1.3, -0.2])
    scale = scale_for_mode_count(parameters, 101, 26)
    predicted = centered_3pl_excess([scale], parameters, 101)[0] + 1

    assert predicted == pytest.approx(26.0)
    assert scale_for_mode_count(parameters, 101, 51) == pytest.approx(1.3)


def test_symmetric_2pl_batch_recovers_golden_parameters():
    scales = np.logspace(-1, 1, 20)
    observed = symmetric_2pl_mode_count(scales, 3.2, 1.4, 101)
    fitted = fit_symmetric_2pl_curves(
        [7], [101], [[0.1, 10.0]], observed[None, :], dataset_name="demo"
    )

    assert fitted.success.tolist() == [True]
    assert fitted.result.parameter_names == ("k", "sigma_half")
    np.testing.assert_allclose(fitted.result.parameters[0], [3.2, 1.4], atol=1e-8)


def test_shape_projection_selects_nearest_curve_samples():
    projected_k, projected_gamma = project_to_shape_curve(
        [1.1, 2.9], [-1.1, 0.8], [-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]
    )

    np.testing.assert_array_equal(projected_k, [1.0, 3.0])
    np.testing.assert_array_equal(projected_gamma, [-1.0, 1.0])


def test_empty_manifold_result_round_trip(tmp_path):
    result = ManifoldAnalysisResult(
        parameter_names=("k", "sigma_half", "log10_gamma"),
        parameters=np.empty((0, 3)),
        frame_ids=np.empty(0, dtype=int),
    )

    restored = ManifoldAnalysisResult.load_npz(result.save_npz(tmp_path / "empty.npz"))

    assert restored.parameters.shape == (0, 3)
    assert restored.frame_ids.size == 0
