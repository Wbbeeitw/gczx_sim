import pandas as pd
import pytest

from revalue_test_utils import install_omegaconf_stub

install_omegaconf_stub()

from rlinf.revalue.evaluation.return_compare import (  # noqa: E402
    ReturnComparisonConfig,
    compare_return_predictions,
)


def test_compare_return_predictions_reports_frame_and_episode_metrics(tmp_path) -> None:
    adv_path = tmp_path / "advantages.parquet"
    pred_path = tmp_path / "predictions.parquet"
    out_path = tmp_path / "report.json"

    pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "return": [-100.0, -50.0, -80.0, -40.0],
            "value_current": [-0.7, -0.3, -0.7, -0.3],
        }
    ).to_parquet(adv_path)
    pd.DataFrame(
        {
            "split": ["train", "train", "val", "val"],
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "value_fused": [-0.9, -0.5, -0.8, -0.4],
        }
    ).to_parquet(pred_path)

    report = compare_return_predictions(
        ReturnComparisonConfig(
            advantages_path=str(adv_path),
            predictions_path=str(pred_path),
            output_path=str(out_path),
            return_min=-100.0,
            return_max=0.0,
        )
    )

    assert out_path.exists()
    assert report["frame_level"]["all"]["rows"] == 4
    assert report["frame_level"]["all"]["episodes"] == 2
    assert report["frame_level"]["all"]["base"]["mse"] == pytest.approx(375.0)
    assert report["frame_level"]["all"]["shared_mlp_fusion"]["mse"] == pytest.approx(
        25.0
    )
    assert report["frame_level"]["all"]["mse_improvement_pct"] == pytest.approx(
        93.3333333
    )
    assert report["episode_level"]["all"]["episodes_improved_mse"] == 2
