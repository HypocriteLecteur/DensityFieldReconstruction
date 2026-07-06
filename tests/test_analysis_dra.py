import numpy as np
import pytest

from dfr.analysis import (
    DRAFrameSamples,
    ManifoldAnalysisResult,
    ModeCurveResult,
    ScaleAnalysisResult,
    fit_design_matrix,
    fit_dra_surface,
    fit_frames,
    fit_one_surface_model,
    mean_nearest_neighbour_distance,
    model_orders,
    select_frames,
)


def make_scale_result(dataset="demo", frame=4, offset=0.0):
    scales = np.array([0.5, 1.0, 2.0, 4.0])
    components = np.array([10, 20, 30, 40])
    orders = components / 200.0
    scale_grid, order_grid = np.meshgrid(scales, orders, indexing="ij")
    coefficients = np.array([-1.2 + offset, 0.35, -0.25])
    dra = 1.0 - np.exp(
        fit_design_matrix(scale_grid.ravel(), order_grid.ravel(), "power")
        @ coefficients
    )
    return ScaleAnalysisResult(
        dataset_name=dataset,
        time_step=frame,
        normalized_scales=scales,
        model_order_percentages=100.0 * orders,
        component_counts=components,
        dra=dra.reshape(scale_grid.shape),
        mean_nnd=1.5,
        number_of_animals=200,
        voxel_res_fraction=0.01,
    )


def test_mode_curve_result_round_trip(tmp_path):
    original = ModeCurveResult(
        scales=[0.25, 0.5, 1.0],
        mode_counts=[12, 7, 2],
        frame=9,
        dataset_name="demo",
    )

    restored = ModeCurveResult.load_npz(original.save_npz(tmp_path / "mode.npz"))

    np.testing.assert_allclose(restored.scales, original.scales)
    np.testing.assert_array_equal(restored.mode_counts, original.mode_counts)
    assert restored.frame == 9 and restored.dataset_name == "demo"


def test_scale_analysis_round_trip_and_legacy_tuple(tmp_path):
    original = make_scale_result()
    restored = ScaleAnalysisResult.load_npz(
        original.save_npz(tmp_path / "surface.npz")
    )

    assert restored.dataset_name == "demo"
    assert restored.is_complete
    np.testing.assert_allclose(restored.scales, original.scales)
    for actual, expected in zip(restored.as_legacy_tuple(), original.as_legacy_tuple()):
        np.testing.assert_allclose(actual, expected)


def test_legacy_scale_cache_requires_missing_context(tmp_path):
    result = make_scale_result()
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        time_step=result.time_step,
        normalized_scales=result.normalized_scales,
        component_numbers=result.component_counts,
        model_order_percentages=result.model_order_percentages,
        mean_nnd=result.mean_nnd,
        dra=result.dra,
        voxel_res_fraction=result.voxel_res_fraction,
    )

    with pytest.raises(ValueError, match="dataset_name"):
        ScaleAnalysisResult.load_npz(path)
    with pytest.raises(ValueError, match="number_of_animals"):
        ScaleAnalysisResult.load_npz(path, dataset_name="demo")
    restored = ScaleAnalysisResult.load_npz(
        path, dataset_name="demo", number_of_animals=200
    )
    assert restored.number_of_animals == 200


def test_manifold_result_round_trip(tmp_path):
    original = ManifoldAnalysisResult(
        parameter_names=("k", "sigma_half"),
        parameters=[[3.0, 0.5], [3.2, 0.7]],
        frame_ids=[10, 20],
        dataset_names=["a", "b"],
    )

    restored = ManifoldAnalysisResult.load_npz(
        original.save_npz(tmp_path / "manifold.npz")
    )

    assert restored.parameter_names == original.parameter_names
    np.testing.assert_allclose(restored.parameters, original.parameters)
    np.testing.assert_array_equal(restored.dataset_names, ["a", "b"])


def test_power_surface_fit_recovers_golden_coefficients():
    result = make_scale_result()
    orders = result.component_counts / result.number_of_animals
    scale_grid, order_grid = np.meshgrid(
        result.normalized_scales, orders, indexing="ij"
    )
    fitted = fit_one_surface_model(
        scale_grid.ravel(), order_grid.ravel(), result.dra.ravel(), "power"
    )
    surface_fit = fit_dra_surface(
        result.normalized_scales,
        result.component_counts,
        result.number_of_animals,
        result.dra,
    )

    np.testing.assert_allclose(fitted["coefficients"], [-1.2, 0.35, -0.25], atol=1e-10)
    assert fitted["rmse"] < 1e-12
    assert surface_fit["candidates"]["power"]["cv_rmse"] < 1e-12


def test_multiframe_fit_uses_shared_samples():
    frames = [
        DRAFrameSamples.from_result(make_scale_result("a", 0)),
        DRAFrameSamples.from_result(make_scale_result("a", 1, 0.02)),
        DRAFrameSamples.from_result(make_scale_result("b", 2, -0.02)),
    ]

    fitted = fit_frames(frames, include_dataset_cv=True)

    assert fitted["best_name"] in {"power", "power_interaction", "log_quadratic"}
    assert all(np.isfinite(candidate["frame_cv_rmse"]) for candidate in fitted["candidates"].values())
    assert all(np.isfinite(candidate["dataset_cv_rmse"]) for candidate in fitted["candidates"].values())


def test_frame_sampling_handles_small_counts_and_bounds():
    np.testing.assert_array_equal(select_frames(10, 20, 1, 14), [14])
    np.testing.assert_array_equal(select_frames(10, 20, 2, 14), [10, 19])
    selected = select_frames(10, 20, 5, 14)
    assert len(selected) == 5
    assert selected[0] == 10 and selected[-1] == 19 and 14 in selected
    with pytest.raises(ValueError, match="positive"):
        select_frames(10, 20, 0, 14)


def test_nnd_and_model_order_validation():
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert mean_nearest_neighbour_distance(points) == 3.0
    components, percentages = model_orders(1000)
    assert len(components) == len(percentages) == 10
    assert len(np.unique(components)) == 10
    with pytest.raises(ValueError, match="At least two"):
        mean_nearest_neighbour_distance(points[:1])
