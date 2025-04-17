"""
Acoustic Simulation Module for SONARES.

This module handles the core acoustic simulation calculations
for the SONARES application.
"""

import numpy as np
from models.source_configurations import SourceConfiguration
from models.material_database import MaterialDatabase
from utils.acoustic_physics import calculate_wave_interference, calculate_energy_density, calculate_material_response, calculate_wavelength

# Import database support
try:
    from models.database import DatabaseHandler
    db_available = True
except ImportError:
    db_available = False

class AcousticSimulation:
    """Class to handle acoustic simulations in a 3D virtual space."""
    
    def __init__(self, room_size=(3, 3, 3), resolution=30):
        """
        Initialize the acoustic simulation.
        
        Args:
            room_size (tuple): Dimensions of the room in meters (x, y, z)
            resolution (int): Number of grid points along each dimension
        """
        self.room_size = np.array(room_size)
        self.resolution = resolution
        self.source_config = SourceConfiguration(room_size)
        self.material_db = MaterialDatabase()
        
        # Generate spatial grid
        self.grid = self._generate_grid()
        
        # Current simulation state
        self.current_material = None
        self.current_frequency = 440.0  # Default frequency (Hz)
        self.current_medium = "air"
        self.current_reflection = 0.0
        self.interference_field = None
        self.resonance_field = None
        
        # Initialize database connection if available
        self.db_available = db_available
        if self.db_available:
            try:
                from models.database import DatabaseHandler
                self.db_handler = DatabaseHandler()
            except Exception as e:
                print(f"Error connecting to database: {e}")
                self.db_available = False
        
    def _generate_grid(self):
        """
        Generate a 3D grid for the simulation.
        
        Returns:
            dict: Dictionary containing X, Y, Z meshgrids
        """
        # Create 1D coordinate arrays
        x = np.linspace(-self.room_size[0]/2, self.room_size[0]/2, self.resolution)
        y = np.linspace(-self.room_size[1]/2, self.room_size[1]/2, self.resolution)
        z = np.linspace(-self.room_size[2]/2, self.room_size[2]/2, self.resolution)
        
        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        return {'X': X, 'Y': Y, 'Z': Z}
    
    def update_grid_resolution(self, resolution):
        """
        Update the grid resolution.
        
        Args:
            resolution (int): New resolution
        """
        self.resolution = resolution
        self.grid = self._generate_grid()
        
    def set_material(self, material_name):
        """
        Set the material for simulation.
        
        Args:
            material_name (str): Name of the material
        """
        self.current_material = material_name
        
    def set_frequency(self, frequency):
        """
        Set the operating frequency.
        
        Args:
            frequency (float): Frequency in Hz
        """
        self.current_frequency = frequency
        self.source_config.set_uniform_frequency(frequency)
        
    def run_simulation(self, medium="air", reflection_coefficient=0.0):
        """
        Run the acoustic simulation with advanced physics.
        
        Args:
            medium (str): Propagation medium (air, water, etc.)
            reflection_coefficient (float): Wall reflection coefficient (0-1)
            
        Returns:
            dict: Simulation results including interference and resonance fields
        """
        # Store current simulation parameters
        self.current_medium = medium
        self.current_reflection = reflection_coefficient
        if not self.source_config.sources:
            return {"error": "No acoustic sources defined"}
        
        if not self.current_material:
            return {"error": "No material selected"}
        
        # Get source properties
        source_props = self.source_config.get_source_properties()
        
        # Calculate wave interference field with advanced physics
        self.interference_field = calculate_wave_interference(
            self.grid, 
            source_props['positions'],
            source_props['frequencies'],
            source_props['phases'],
            source_props['amplitudes'],
            medium=medium,
            reflection_coefficient=reflection_coefficient
        )
        
        # Calculate material resonance response
        resonance_factor = self.material_db.calculate_resonance_response(
            self.current_material, 
            self.current_frequency
        )
        
        # Get material properties for material-specific calculations
        material_props = self.material_db.get_material_properties(self.current_material)
        
        # Combine interference field with material response
        self.resonance_field = self.interference_field * resonance_factor
        
        # Calculate material absorption effect (more realistic material physics)
        if material_props:
            # Damping affects how quickly energy is absorbed by the material
            damping = material_props["damping_coefficient"]
            elasticity = material_props["elasticity"]
            
            # Adjust resonance field based on material properties
            # Higher damping = more absorption, higher elasticity = less absorption
            absorption_factor = damping / (elasticity + 0.1)  # Avoid division by near-zero
            
            # Apply non-linear effects at high intensities (saturation)
            saturation_threshold = 0.7
            high_intensity_mask = self.resonance_field > saturation_threshold
            
            if np.any(high_intensity_mask):
                # Apply saturation effect to high-intensity regions
                saturated_values = saturation_threshold + (1.0 - saturation_threshold) * \
                                   (1.0 - np.exp(-(self.resonance_field[high_intensity_mask] - saturation_threshold) / 0.2))
                self.resonance_field[high_intensity_mask] = saturated_values
        
        return {
            "interference_field": self.interference_field,
            "resonance_field": self.resonance_field,
            "resonance_factor": resonance_factor,
            "medium": medium,
            "reflection_coefficient": reflection_coefficient
        }
    
    def get_2d_slice(self, axis='z', position=0):
        """
        Get a 2D slice of the 3D simulation field.
        
        Args:
            axis (str): Axis perpendicular to the slice ('x', 'y', or 'z')
            position (float): Position along the axis (-1.5 to 1.5)
            
        Returns:
            dict: 2D slice data for visualization
        """
        if self.interference_field is None:
            return None
        
        # Convert position to index
        pos_fraction = (position + self.room_size[0]/2) / self.room_size[0]
        idx = int(pos_fraction * (self.resolution - 1))
        idx = max(0, min(idx, self.resolution - 1))  # Clamp to valid range
        
        # Extract 2D slice based on axis
        if axis == 'x':
            i_field_slice = self.interference_field[idx, :, :]
            r_field_slice = self.resonance_field[idx, :, :]
            x_grid, y_grid = self.grid['Y'][idx, :, :], self.grid['Z'][idx, :, :]
            axes_labels = ('Y', 'Z')
        elif axis == 'y':
            i_field_slice = self.interference_field[:, idx, :]
            r_field_slice = self.resonance_field[:, idx, :]
            x_grid, y_grid = self.grid['X'][:, idx, :], self.grid['Z'][:, idx, :]
            axes_labels = ('X', 'Z')
        else:  # axis == 'z'
            i_field_slice = self.interference_field[:, :, idx]
            r_field_slice = self.resonance_field[:, :, idx]
            x_grid, y_grid = self.grid['X'][:, :, idx], self.grid['Y'][:, :, idx]
            axes_labels = ('X', 'Y')
            
        return {
            'interference': i_field_slice,
            'resonance': r_field_slice,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'axes_labels': axes_labels
        }
    
    def analyze_hotspots(self, threshold=0.7):
        """
        Analyze resonance hotspots in the material.
        
        Args:
            threshold (float): Threshold value for hotspot detection (0-1)
            
        Returns:
            dict: Hotspot analysis results
        """
        if self.resonance_field is None:
            return None
            
        # Find points above threshold
        hotspots = np.where(self.resonance_field > threshold)
        
        if len(hotspots[0]) == 0:
            return {
                'count': 0,
                'max_intensity': 0.0,
                'positions': [],
                'intensities': []
            }
            
        # Get coordinates and values at hotspots
        x_coords = self.grid['X'][hotspots]
        y_coords = self.grid['Y'][hotspots]
        z_coords = self.grid['Z'][hotspots]
        intensities = self.resonance_field[hotspots]
        
        # Combine coordinates into position tuples
        positions = np.column_stack((x_coords, y_coords, z_coords))
        
        return {
            'count': len(intensities),
            'max_intensity': np.max(intensities),
            'positions': positions,
            'intensities': intensities
        }
    
    def get_frequency_response(self, freq_range=(20, 2000), steps=100):
        """
        Calculate material's frequency response over a range.
        
        Args:
            freq_range (tuple): Frequency range (min, max) in Hz
            steps (int): Number of frequency steps
            
        Returns:
            dict: Frequency response data
        """
        if not self.current_material:
            return None
            
        frequencies = np.linspace(freq_range[0], freq_range[1], steps)
        responses = []
        
        for freq in frequencies:
            response = self.material_db.calculate_resonance_response(
                self.current_material, 
                freq
            )
            responses.append(response)
            
        return {
            'frequencies': frequencies,
            'responses': np.array(responses)
        }
        
    def save_simulation_to_database(self, name, notes=None):
        """
        Save the current simulation results to the database.
        
        Args:
            name (str): Name for this simulation record
            notes (str, optional): Additional notes about this simulation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.db_available:
            return False
            
        if self.resonance_field is None or self.current_material is None:
            return False
            
        try:
            # Calculate hotspots for database record
            hotspot_data = self.analyze_hotspots(threshold=0.7)
            hotspot_count = hotspot_data['count'] if hotspot_data else 0
            
            # Get source arrangement info
            source_count = len(self.source_config.sources)
            
            # Max intensity
            max_intensity = float(np.max(self.resonance_field))
            
            # Create simulation record
            sim_data = {
                'name': name,
                'material_name': self.current_material,
                'frequency': float(self.current_frequency),
                'source_count': source_count,
                'source_arrangement': 'custom',  # This could be improved
                'medium': self.current_medium,
                'reflection_coefficient': float(self.current_reflection),
                'max_intensity': max_intensity,
                'resonance_factor': float(max_intensity),
                'resonance_hotspots': hotspot_count,
                'notes': notes
            }
            
            self.db_handler.save_simulation_result(sim_data)
            return True
        except Exception as e:
            print(f"Error saving to database: {e}")
            return False
            
    def save_experimental_data_to_database(self, name, data_df, notes=None):
        """
        Save experimental data to the database.
        
        Args:
            name (str): Name for this dataset
            data_df (pd.DataFrame): Pandas DataFrame with experimental data
            notes (str, optional): Additional notes
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.db_available:
            return False
            
        if self.current_material is None or data_df is None:
            return False
            
        try:
            self.db_handler.save_experimental_data(
                name=name,
                data_df=data_df,
                material_name=self.current_material,
                notes=notes
            )
            return True
        except Exception as e:
            print(f"Error saving experimental data to database: {e}")
            return False
