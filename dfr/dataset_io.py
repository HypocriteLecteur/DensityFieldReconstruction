import numpy as np
import scipy.io as sio
from abc import ABC, abstractmethod
import os
import h5py
import pandas as pd

# --- 1. Define the Standardized Data Format ---
# Position: (number_of_frames, number_of_agents, 3)
# Velocity: (number_of_frames, number_of_agents, 3)

class InvalidFileFormatError(Exception):
    """Custom exception for when a loader cannot parse a file."""
    pass

# --- 2. Create Velocity Calculation Strategy Interface ---
class VelocityStrategy(ABC):
    """Interface for different velocity calculation methods."""
    @abstractmethod
    def calculate(self, positions: np.ndarray) -> np.ndarray:
        """Calculates velocities from position data."""
        pass

class ForwardDifference(VelocityStrategy):
    """Calculates velocity using a simple forward difference method."""
    def calculate(self, positions: np.ndarray) -> np.ndarray:
        # Assumes positions shape: (n_frames, n_agents, 3)
        velocities = np.zeros_like(positions)
        # Compute forward difference for all but the last frame
        velocities[:-1, :, :] = positions[1:, :, :] - positions[:-1, :, :]
        # For the last frame, use backward difference as an approximation
        velocities[-1, :, :] = velocities[-2, :, :]
        return velocities

class CentralDifference(VelocityStrategy):
    """Calculates velocity using the more accurate central difference method."""
    def calculate(self, positions: np.ndarray) -> np.ndarray:
        velocities = np.zeros_like(positions)
        # Central difference for interior points0
        velocities[1:-1] = (positions[2:] - positions[:-2]) / 2.0
        # One-sided (forward) difference for the first point
        velocities[0] = positions[1] - positions[0]
        # One-sided (backward) difference for the last point
        velocities[-1] = positions[-1] - positions[-2]
        return velocities

# --- 3. Update the Dataset Interface ---
class DatasetInterface(ABC):
    """
    Defines the contract for all data loaders, now including velocity handling.
    """
    def __init__(self):
        self._trajectory_data = None
        self._velocity_data = None
    
    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    @property
    def trajectories(self) -> np.ndarray:
        if self._trajectory_data is None:
            raise ValueError("Data has not been loaded. Call load() first.")
        return self._trajectory_data

    @property
    def velocities(self) -> np.ndarray:
        if self._velocity_data is None:
            raise ValueError("Velocity data is not available or has not been calculated.")
        return self._velocity_data

    def positions_at_time_step(self, time_step) -> np.ndarray:
        if self._trajectory_data is None:
            raise ValueError("Data has not been loaded. Call load() first.")
        positions = self.trajectories[time_step]
        mask = ~np.isnan(positions[:, 0])
        return positions[mask]
    
    def positions_at_time_step_mask(self, time_step) -> np.ndarray:
        if self._trajectory_data is None:
            raise ValueError("Data has not been loaded. Call load() first.")
        positions = self.trajectories[time_step]
        mask = ~np.isnan(positions[:, 0])
        return positions[mask], mask

    def calculate_velocities(self, strategy: VelocityStrategy):
        """
        Calculates velocities from trajectories if they don't already exist.
        This method is idempotent; it won't recalculate if velocities are present.
        """
        if self._velocity_data is not None:
            print("  -> Velocity data already exists. Skipping calculation.")
            return
        
        print(f"  -> Calculating velocities using '{strategy.__class__.__name__}'...")
        if self._trajectory_data is None:
            raise ValueError("Cannot calculate velocities before loading trajectory data.")
        
        self._velocity_data = strategy.calculate(self._trajectory_data)


# --- 4. Update Concrete Loader Classes ---

class NpyLoader(DatasetInterface):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.npy',)
    
    """Loads a standard .npy file (assumed position only)."""
    def load(self, filepath: str) -> None:
        try:
            data = np.load(filepath)
            if data.ndim != 3 or data.shape[2] != 3:
                raise InvalidFileFormatError("NPY data is not in (frames, agents, 3) format.")
            self._trajectory_data = data
            # No velocity data in this format
        except Exception as e:
            raise InvalidFileFormatError(f"Failed to load NPY file: {e}") from e

class NpzStandardLoader(DatasetInterface):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.npz',)

    """Loads .npz files with 'trajectories' and optional 'velocities' keys."""
    def load(self, filepath: str) -> None:
        try:
            with np.load(filepath) as data:
                # Load trajectories (mandatory)
                trajectories = data['trajectories']
                if trajectories.ndim != 3 or trajectories.shape[2] != 3:
                     raise InvalidFileFormatError("NPZ 'trajectories' is not in (frames, agents, 3) format.")
                self._trajectory_data = trajectories

                # Load velocities (optional)
                if 'velocities' in data:
                    velocities = data['velocities']
                    if velocities.shape != trajectories.shape:
                        raise InvalidFileFormatError("Velocities shape must match trajectories shape.")
                    self._velocity_data = velocities

        except KeyError as e:
            raise InvalidFileFormatError("File does not contain required 'trajectories' key.") from e
        except Exception as e:
            raise InvalidFileFormatError(f"Failed to load standard NPZ file: {e}") from e

class NpzPositionsLoader(DatasetInterface):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.npz',)

    """Loads .npz files where data is under 'positions' and needs reshaping (position only)."""
    def load(self, filepath: str) -> None:
        try:
            with np.load(filepath) as data:
                positions = data['positions']
                if positions.ndim != 3 or positions.shape[2] != 3:
                    raise InvalidFileFormatError("NPZ 'positions' key is not in a 3D format.")
                self._trajectory_data = positions
                # No velocity data in this format
        except KeyError as e:
            raise InvalidFileFormatError("File does not contain 'positions' key.") from e
        except Exception as e:
            raise InvalidFileFormatError(f"Failed to load positions NPZ file: {e}") from e

class MatSwarmLoader(DatasetInterface):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.mat',)

    """Loads .mat files with 'swarm_data.positions' (position only)."""
    def load(self, filepath: str) -> None:
        try:
            mat_data = sio.loadmat(filepath)
            positions = mat_data['swarm_data']['positions'][0, 0]
            if positions.ndim != 3 or positions.shape[0] != 3:
                 raise InvalidFileFormatError("MAT 'positions' is not in (3, agents, frames) format.")
            self._trajectory_data = np.transpose(positions, (2, 1, 0))
            # No velocity data in this format
        except KeyError as e:
            raise InvalidFileFormatError("MAT file structure does not match 'swarm_data.positions'.") from e
        except Exception as e:
            raise InvalidFileFormatError(f"Failed to load MAT file: {e}") from e

class RtfLoader(DatasetInterface):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.rtf',)

    """Loads .rtf files with custom parsing (position only)."""
    def load(self, filepath: str) -> None:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            data_points = []
            parsing_data = False

            for line in lines:
                cleaned_line = line.strip().replace('\\', '').replace('}', '')

                # We look for the line that indicates the start of the data block.
                if '#  x(t1)    y(t1)    z(t1)      x(t2)  y(t2)    z(t2)' in cleaned_line:
                    parsing_data = True
                    continue
                
                if parsing_data:
                    if cleaned_line.startswith('#') or not cleaned_line:
                        continue
                    
                    parts = cleaned_line.split()
                    if len(parts) < 6:
                        continue
                    
                    try:
                        point_t1 = list(map(float, parts[0:3]))
                        point_t2 = list(map(float, parts[3:6]))
                        data_points.append(point_t1)
                        data_points.append(point_t2)
                    except ValueError:
                        continue
            
            if not data_points:
                raise InvalidFileFormatError("No valid data points found in RTF file.")
            
            data_array = np.array(data_points)
            n_agents = data_array.shape[0] // 2
            n_frames = 2  # Since each frame has two points (t1 and t2)
            self._trajectory_data = data_array.reshape((n_agents, n_frames, 3)).transpose((1, 0, 2))
            # No velocity data in this format

        except Exception as e:
            raise InvalidFileFormatError(f"Failed to load RTF file: {e}") from e

class Hdf5Loader(DatasetInterface):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.hdf5',)

    def _get_cache_paths(self, filepath: str) -> tuple[str, str]:
        # Split into two .npy files so they can be memory-mapped
        return filepath + ".traj.cache.npy", filepath + ".vel.cache.npy"

    def load(self, filepath: str) -> None:
        traj_cache, vel_cache = self._get_cache_paths(filepath)
        
        # 1. Check if a valid cache exists
        if os.path.exists(traj_cache) and os.path.exists(vel_cache):
            source_mtime = os.path.getmtime(filepath)
            cache_mtime = min(os.path.getmtime(traj_cache), os.path.getmtime(vel_cache))
            
            # Only load if the cache is newer than the source file
            if cache_mtime > source_mtime:
                print(f"  -> Loading from cache: {os.path.basename(traj_cache)} & vel cache")
                
                # mmap_mode='r' is the magic here. It loads pointers, not data.
                self._trajectory_data = np.load(traj_cache, mmap_mode='r')
                self._velocity_data = np.load(vel_cache, mmap_mode='r')
                return
            
        # 2. If no cache, perform the heavy processing using memmap to save RAM
        try:
            print(f"  -> Processing raw HDF5: {os.path.basename(filepath)}")
            with h5py.File(filepath, 'r') as f:
                timestamps_str = sorted(f.keys(), key=lambda x: int(x))
                n_frames = len(timestamps_str)
                
                # Identify unique agents
                all_tids = set()
                for t_str in timestamps_str:
                    all_tids.update(f[t_str]['tid'][:])
                
                sorted_tids = sorted(list(all_tids))
                tid_to_idx = {tid: i for i, tid in enumerate(sorted_tids)}
                n_agents = len(sorted_tids)

                # Initialize memmapped arrays on disk instead of in RAM
                self._trajectory_data = np.lib.format.open_memmap(
                    traj_cache, mode='w+', dtype=np.float64, shape=(n_frames, n_agents, 3)
                )
                self._velocity_data = np.lib.format.open_memmap(
                    vel_cache, mode='w+', dtype=np.float64, shape=(n_frames, n_agents, 3)
                )

                # Fill the disk-backed arrays with NaNs safely
                self._trajectory_data[:] = np.nan
                self._velocity_data[:] = np.nan

                for frame_idx, t_str in enumerate(timestamps_str):
                    group = f[t_str]
                    tids = group['tid'][:]
                    
                    indices = [tid_to_idx[tid] for tid in tids]
                    
                    # Writing to these bulk assigns directly to the disk cache
                    self._trajectory_data[frame_idx, indices, 0] = group['x'][:]
                    self._trajectory_data[frame_idx, indices, 1] = group['y'][:]
                    self._trajectory_data[frame_idx, indices, 2] = group['z'][:]
                    
                    self._velocity_data[frame_idx, indices, 0] = group['vx'][:]
                    self._velocity_data[frame_idx, indices, 1] = group['vy'][:]
                    self._velocity_data[frame_idx, indices, 2] = group['vz'][:]

            # Flush the memory-mapped arrays to ensure data is written to disk
            self._trajectory_data.flush()
            self._velocity_data.flush()

            # Re-open in read-only mode to prevent accidental overwrites later
            self._trajectory_data = np.load(traj_cache, mmap_mode='r')
            self._velocity_data = np.load(vel_cache, mmap_mode='r')
            
            print(f"  -> Caches created: {os.path.basename(traj_cache)}, {os.path.basename(vel_cache)}")

        except Exception as e:
            # Clean up partial cache files if something fails
            if os.path.exists(traj_cache): os.remove(traj_cache)
            if os.path.exists(vel_cache): os.remove(vel_cache)
            raise InvalidFileFormatError(f"Failed to load HDF5 file: {e}") from e

def load_camera_extrinsics(filepath: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Loads camera extrinsics from a CSV file with multi-row structured format.
    
    Expected format:
    - Each time step spans 3 rows (one for each row of the 3x3 rotation matrix)
    - For each camera: position (Pos_X, Pos_Y, Pos_Z) and rotation matrix (R_ij)
    
    Args:
        filepath: Path to the camera_extrinsics.csv file
    
    Returns:
        tuple: (R_list, T_list) where:
            - R_list: List of dicts {camera_id: rotation_matrix (3x3 ndarray)}
            - T_list: List of dicts {camera_id: translation_vector (3,) ndarray}
    """
    try:
        # Read CSV without automatic header parsing
        raw_data = pd.read_csv(filepath, header=None)
        raw_data.fillna('', inplace=True)  # Fill NaN with empty strings
        
        # Extract header row (row 0)
        headers = raw_data.iloc[0].tolist()
        
        # Find camera columns and their structure
        cameras = {}
        for i, header in enumerate(headers):
            if header.startswith('Camera'):
                parts = header.split('_')
                camera_id = parts[0]  # e.g., "Camera1"
                if camera_id not in cameras:
                    cameras[camera_id] = {'pos_cols': [], 'rot_cols': []}
                
                if 'Pos' in header:
                    cameras[camera_id]['pos_cols'].append((header, i))
                elif 'R' in header:
                    cameras[camera_id]['rot_cols'].append((header, i))
        
        # Parse data rows (skip header rows 0-2)
        r_list = []
        t_list = []
        
        row_idx = 3
        while row_idx < len(raw_data):
            time_val = raw_data.iloc[row_idx, 0]
            
            # Skip if this is not a time row (should have numeric first column)
            if pd.isna(time_val) or time_val == '':
                row_idx += 1
                continue
            
            # Get the 3 rows that comprise this time step
            if row_idx + 2 >= len(raw_data):
                break
            
            rows = [
                raw_data.iloc[row_idx].tolist(),
                raw_data.iloc[row_idx + 1].tolist(),
                raw_data.iloc[row_idx + 2].tolist()
            ]
            
            # Extract data for this time step
            R_dict = {}
            T_dict = {}
            
            for camera_id in sorted(cameras.keys()):
                # Extract position (T)
                T = np.zeros(3)
                for header, col_idx in cameras[camera_id]['pos_cols']:
                    coord_idx = int(header[-1])  # Get X(0), Y(1), Z(2) from header
                    if col_idx < len(rows[coord_idx]):
                        try:
                            T[coord_idx] = float(rows[coord_idx][col_idx])
                        except (ValueError, TypeError):
                            pass
                
                # Extract rotation matrix (R)
                R = np.zeros((3, 3))
                for header, col_idx in cameras[camera_id]['rot_cols']:
                    # Parse R_ij from header
                    r_part = header.split('_')[1]  # e.g., "00", "01", etc.
                    i, j = int(r_part[0]), int(r_part[1])
                    if col_idx < len(rows[i]):
                        try:
                            R[i, j] = float(rows[i][col_idx])
                        except (ValueError, TypeError):
                            pass
                
                R_dict[camera_id] = R
                T_dict[camera_id] = T
            
            r_list.append(R_dict)
            t_list.append(T_dict)
            
            row_idx += 3
        
        return r_list, t_list
    
    except Exception as e:
        raise InvalidFileFormatError(f"Failed to load camera extrinsics CSV: {e}") from e


class CsvDronePositionLoader(DatasetInterface):
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return ('.csv',)
    
    """
    Loads .csv files with drone position data.
    Expected format: Time column followed by Drone##_X, Drone##_Y, Drone##_Z columns for each drone.
    Converts to standardized format: (n_frames, n_agents, 3)
    """
    def load(self, filepath: str) -> None:
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Extract time column
            if 'Time' not in df.columns:
                raise InvalidFileFormatError("CSV file must contain 'Time' column.")
            
            # Parse drone columns dynamically
            drone_data = self._extract_drone_positions(df)
            
            if not drone_data:
                raise InvalidFileFormatError("No drone position data found in CSV file.")
            
            # Convert to standardized format: (n_frames, n_agents, 3)
            n_frames = len(df)
            n_agents = len(drone_data)
            
            self._trajectory_data = np.zeros((n_frames, n_agents, 3))
            
            for agent_idx, (drone_id, positions) in enumerate(sorted(drone_data.items())):
                self._trajectory_data[:, agent_idx, :] = positions[:, [1, 0, 2]] # coordinate frame conversion, left hand to right hand
            
            # No velocity data in this format
            
        except Exception as e:
            raise InvalidFileFormatError(f"Failed to load CSV file: {e}") from e
    
    def _extract_drone_positions(self, df: pd.DataFrame) -> dict:
        """
        Extracts drone position data from DataFrame.
        
        Returns:
            dict: {drone_id: positions_array} where positions_array is (n_frames, 3)
        """
        drone_data = {}

        # Filter columns that contain drone coordinates
        drone_cols = [c for c in df.columns if c.startswith('Drone') and c[-2:] in ['_X', '_Y', '_Z']]
        
        # Group by the drone prefix (e.g., 'Drone00')
        prefixes = set(c.rsplit('_', 1)[0] for c in drone_cols)
        
        for prefix in sorted(prefixes):
            cols = [f"{prefix}_X", f"{prefix}_Y", f"{prefix}_Z"]
            if all(col in df.columns for col in cols):
                # Directly extract the underlying numpy array
                drone_data[prefix] = df[cols].to_numpy()
        
        # # Find all unique drone IDs by parsing column names
        # drone_ids = set()
        # for col in df.columns:
        #     if col.startswith('Drone') and any(c in col for c in ['_X', '_Y', '_Z']):
        #         # Extract drone ID from column name like "Drone00_X"
        #         drone_id = col.split('_')[0]  # e.g., "Drone00"
        #         drone_ids.add(drone_id)

        
        # # For each drone, extract X, Y, Z coordinates
        # for drone_id in sorted(drone_ids):
        #     x_col = f"{drone_id}_X"
        #     y_col = f"{drone_id}_Y"
        #     z_col = f"{drone_id}_Z"
            
        #     if all(col in df.columns for col in [x_col, y_col, z_col]):
        #         positions = np.column_stack([
        #             df[x_col].values,
        #             df[y_col].values,
        #             df[z_col].values
        #         ])
        #         drone_data[drone_id] = positions
        
        return drone_data

# --- 5. Create the Dataset Factory ---
class DatasetFactory:
    """
    Intelligently provides the correct data loader for a given file.
    This factory automatically discovers all available loaders.
    """
    def __init__(self, verbose: bool = False):
        self._loaders = []
        self.verbose = verbose
        self.discover_loaders()

    def discover_loaders(self):
        """
        Finds and registers all concrete subclasses of DatasetInterface.
        """
        if self.verbose:
            print("Discovering available loaders...")

        # Get all subclasses recursively to support more complex inheritance
        all_subclasses = []
        def get_all_subclasses(cls):
            for subclass in cls.__subclasses__():
                all_subclasses.append(subclass)
                get_all_subclasses(subclass)
        get_all_subclasses(DatasetInterface)
        
        for loader_class in all_subclasses:
            self.register_loader(loader_class)

    def register_loader(self, loader_class: type[DatasetInterface]):
        """
        Registers a loader class.
        """
        self._loaders.append(loader_class)
        if self.verbose:
            print(f"  - Registered '{loader_class.__name__}'")

    def get_dataset(self, filepath: str) -> DatasetInterface:
        """
        Tries all registered loaders on the file until one succeeds.
        """
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        # Filter loaders by extension
        valid_loaders = [l for l in self._loaders if ext in l().supported_extensions]

        for loader_class in valid_loaders:
            try:
                loader_instance = loader_class()
                loader_instance.load(filepath)
                return loader_instance
            except InvalidFileFormatError:
                continue
                
        raise ValueError(f"Could not find a suitable loader for the file: {filepath}")