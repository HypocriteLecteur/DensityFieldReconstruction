import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from dfr import EvaluationConfig
from dfr.evaluation import EvaluationRun, EvaluationSummary, FrameEvaluation
from dfr.plotting import plot_evaluation_metric_series, plot_evaluation_summary


def _summary(
    true_positive=8.0,
    false_positive=1.0,
    false_negative=2.0,
    ground_truth=10.0,
    predicted=9.0,
) -> EvaluationSummary:
    return EvaluationSummary(
        true_positive,
        false_positive,
        false_negative,
        ground_truth,
        predicted,
    )


def _frame(frame=0) -> FrameEvaluation:
    return FrameEvaluation(
        dataset_name="tiny",
        frame=frame,
        summary=_summary(),
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        voxel_resolution=0.25,
    )


def _run() -> EvaluationRun:
    return EvaluationRun(
        frames=(_frame(0), _frame(1)),
        config=EvaluationConfig(device="cpu"),
    )


def test_plot_evaluation_summary_accepts_evaluation_run():
    fig, ax = plot_evaluation_summary(_run())

    assert fig is ax.figure
    assert ax.get_title() == "Evaluation summary (2 frame(s))"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "Recall",
        "Miss",
        "Hallucination",
        "dMOTA",
    ]
    assert len(ax.patches) == 4
    plt.close(fig)


def test_plot_evaluation_metric_series_accepts_evaluation_run():
    fig, ax = plot_evaluation_metric_series(_run())

    assert fig is ax.figure
    assert ax.get_title() == "Evaluation metrics by frame"
    assert ax.get_xlabel() == "Frame"
    assert [line.get_label() for line in ax.lines] == [
        "Recall",
        "Hallucination",
        "dMOTA",
    ]
    plt.close(fig)


def test_plot_evaluation_metric_series_accepts_frame_sequence_and_existing_axis():
    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_evaluation_metric_series(
        (_frame(3), _frame(5)),
        metrics=("miss", "dmota"),
        ax=ax,
        title="Frame metrics",
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert ax.get_title() == "Frame metrics"
    assert [line.get_label() for line in ax.lines] == ["Miss", "dMOTA"]
    assert ax.lines[0].get_xdata().tolist() == [3, 5]
    plt.close(fig)


def test_plot_evaluation_summary_accepts_frame_evaluation_and_metric_subset():
    fig, ax = plot_evaluation_summary(_frame(7), metrics=("recall", "dmota"))

    assert ax.get_title() == "tiny frame 7 evaluation"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Recall", "dMOTA"]
    assert len(ax.patches) == 2
    plt.close(fig)


def test_plot_evaluation_summary_accepts_bare_summary_and_existing_axis():
    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_evaluation_summary(
        _summary(),
        metrics=("hallucination",),
        ax=ax,
        title="Custom evaluation",
        ylim=None,
    )

    assert returned_fig is fig
    assert returned_ax is ax
    assert ax.get_title() == "Custom evaluation"
    assert len(ax.patches) == 1
    plt.close(fig)


def test_plot_evaluation_summary_validates_inputs():
    with pytest.raises(ValueError, match="at least one"):
        plot_evaluation_summary(_summary(), metrics=())
    with pytest.raises(ValueError, match="Unsupported"):
        plot_evaluation_summary(_summary(), metrics=("precision",))
    with pytest.raises(TypeError, match="EvaluationRun"):
        plot_evaluation_summary(object())


def test_plot_evaluation_metric_series_validates_inputs():
    with pytest.raises(ValueError, match="at least one"):
        plot_evaluation_metric_series(())
    with pytest.raises(ValueError, match="Unsupported"):
        plot_evaluation_metric_series(_run(), metrics=("precision",))
    with pytest.raises(TypeError, match="FrameEvaluation"):
        plot_evaluation_metric_series([object()])
