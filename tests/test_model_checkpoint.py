from types import SimpleNamespace

import torch

from dfr.model_checkpoint import build_checkpoint, load_training_history, save_history, write_checkpoints


def make_model():
    model = SimpleNamespace()
    model._xyz = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]]))
    model._radius = torch.nn.Parameter(torch.tensor([[0.5]]))
    model._weights = torch.nn.Parameter(torch.tensor([[1.0]]))
    model.optimizer = torch.optim.Adam([model._xyz, model._radius, model._weights])
    model.xyz_reg = 0.1
    model.radius_reg = 0.2
    model.radius_cutoff_inv = 10.0
    model.xyz_lr_c = 1e-2
    model.xyz_lr_final_c = 1e-3
    model.radius_lr_c = 2e-2
    model.radius_lr_final_c = 2e-3
    model.weights_lr_c = 3e-2
    model.weights_lr_final_c = 3e-3
    model.optimizer_type = "Adam"
    model.rasterizer_h = 64
    model.rasterizer_w = 64
    model.rasterizer_p_max = 8
    model.rasterizer_list = [object(), object()]
    return model


def test_checkpoint_build_and_file_round_trip(tmp_path):
    checkpoint = build_checkpoint(make_model())
    checkpoint_path = tmp_path / "checkpoints" / "history.pth"
    metrics_path = tmp_path / "metrics" / "history.pth"

    write_checkpoints([checkpoint], str(checkpoint_path))
    save_history([{"loss": 1.25}], str(metrics_path))
    restored = load_training_history(str(checkpoint_path), device="cpu")
    metrics = load_training_history(str(metrics_path), device="cpu")

    assert checkpoint_path.exists()
    assert metrics_path.exists()
    assert restored[0]["cam_num"] == 2
    assert restored[0]["rasterizer_h"] == 64
    torch.testing.assert_close(restored[0]["_xyz"], torch.tensor([[1.0, 2.0, 3.0]]))
    assert metrics == [{"loss": 1.25}]
