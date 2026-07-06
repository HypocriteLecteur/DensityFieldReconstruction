from dfr.config import ReconstructionParams, TrainingParams


def test_typed_configs_round_trip_dicts():
    training_values = {
        "xyz_lr_c": 0.1,
        "xyz_lr_final_c": 0.01,
        "radius_lr_c": 0.2,
        "radius_lr_final_c": 0.02,
        "weights_lr_c": 0.3,
        "weights_lr_final_c": 0.03,
        "xyz_reg": 1.0,
        "radius_reg": 2.0,
        "radius_cutoff_inv": 10.0,
        "lr_max_steps": 100,
    }
    reconstruction_values = {
        "targetd_num_mode": 10,
        "voxel_scale": 0.5,
        "voxel_peak_threshold": 0.2,
        "voxel_grid_max_size": 128,
        "voxel_peaks_number": 20,
    }

    assert TrainingParams.from_dict(training_values).to_dict() == training_values
    assert ReconstructionParams.from_dict(reconstruction_values).to_dict() == reconstruction_values
