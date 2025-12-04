import numpy as np
import scipy.io as sio
from abc import ABC, abstractmethod
import os

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

# --- 5. Create the Dataset Factory (Unchanged) ---
class DatasetFactory:
    """
    Intelligently provides the correct data loader for a given file.
    This factory automatically discovers all available loaders.
    """
    def __init__(self):
        self._loaders = []
        self.discover_loaders()

    def discover_loaders(self):
        """
        Finds and registers all concrete subclasses of DatasetInterface.
        """
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
        print(f"  - Registered '{loader_class.__name__}'")

    def get_dataset(self, filepath: str) -> DatasetInterface:
        """
        Tries all registered loaders on the file until one succeeds.
        """
        # Prioritize loaders that might handle more complex files first.
        # Simple heuristic: class name length (longer names are often more specific)
        sorted_loaders = sorted(self._loaders, key=lambda x: len(x.__name__), reverse=True)

        for loader_class in sorted_loaders:
            try:
                loader_instance = loader_class()
                loader_instance.load(filepath)
                print(f"Successfully loaded '{os.path.basename(filepath)}' using '{loader_class.__name__}'")
                return loader_instance
            except InvalidFileFormatError:
                continue
            except Exception:
                continue
                
        raise ValueError(f"Could not find a suitable loader for the file: {filepath}")