from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.evaluation.return_compare import (  # noqa: E402
    summarize_return_report,
)


def test_summary_extracts_frame_and_episode_rows() -> None:
    report = {
        "frame_level": {
            "all": {
                "rows": 4,
                "episodes": 2,
                "base": {"mse": 100.0, "rmse": 10.0, "mae": 8.0, "bias": -2.0},
                "shared_mlp_fusion": {
                    "mse": 81.0,
                    "rmse": 9.0,
                    "mae": 7.0,
                    "bias": -1.0,
                },
                "mse_improvement_pct": 19.0,
                "rmse_improvement_pct": 10.0,
                "mae_improvement_pct": 12.5,
                "bias_abs_improvement_pct": 50.0,
            }
        },
        "episode_level": {
            "all": {
                "episodes": 2,
                "episodes_improved_mse": 1,
                "mean_base_mse": 100.0,
                "mean_fused_mse": 81.0,
                "mean_mse_gain": 19.0,
                "median_mse_gain": 19.0,
                "mean_base_mae": 8.0,
                "mean_fused_mae": 7.0,
                "mean_mae_gain": 1.0,
            }
        },
    }

    summary = summarize_return_report(report)
    frame_rows = summary["frame_rows"]
    episode_rows = summary["episode_rows"]

    assert len(frame_rows) == 1
    assert frame_rows[0]["split"] == "all"
    assert frame_rows[0]["mse_improvement_pct"] == 19.0
    assert frame_rows[0]["mae_improvement_pct"] == 12.5

    assert len(episode_rows) == 1
    assert episode_rows[0]["split"] == "all"
    assert episode_rows[0]["episodes_improved_mse"] == 1
    assert episode_rows[0]["episodes_improved_mse_pct"] == 50.0
