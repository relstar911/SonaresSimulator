"""
Acoustic Simulation Module for SONARES.

This module handles the core acoustic simulation calculations
for the SONARES application.
"""

import numpy as np
from models.source_configurations import SourceConfiguration
from models.material_database import MaterialDatabase
from utils.acoustic_physics import calculate_wave_interference

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
        self.interference_field = None
        self.resonance_field = None
        
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
        
    def run_simulation(self):
        """
        Run the acoustic simulation.
        
        Returns:
            dict: Simulation results including interference and resonance fields
        """
        if not self.source_config.sources:
            return {"error": "No acoustic sources defined"}
        
        if not self.current_material:
            return {"error": "No material selected"}
        
        # Get source properties
        source_props = self.source_config.get_source_properties()
        
        # Calculate wave interference field
        self.interference_field = calculate_wave_interference(
            self.grid, 
            source_props['positions'],
            source_props['frequencies'],
            source_props['phases'],
            source_props['amplitudes']
        )
        
        # Calculate material resonance response
        resonance_factor = self.material_db.calculate_resonance_response(
            self.current_material, 
            self.current_frequency
        )
        
        # Combine interference field with material response
        self.resonance_field = self.interference_field * resonance_factor
        
        return {
            "interference_field": self.interference_field,
            "resonance_field": self.resonance_field,
            "resonance_factor": resonance_factor
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
