import yaml
import numpy as np

class SimulationConfig:
    """Holds configuration parameters for the simulation."""
    def __init__(self, config_file):
        self.load_from_file(config_file)

    def load_from_file(self, config_file):
        """Load configuration from a YAML file."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.cam_pose = np.array(config['cam_pose'])
            self.cam_pose2 = np.array(config['cam_pose2'])
            self.intrinsics_params = np.array(config['intrinsics_params'])
            self.H = config['H']
            self.W = config['W']
            self.iter = config['iter']
            self.near_clip = config['near_clip']
            self.far_clip = config['far_clip']
            self.size = config['size']
            self.save_video = config['save_video']
            self.fps = config['fps']
            self.dpi = config['dpi']
            self.data_file = config['data_file']