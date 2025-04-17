"""
Material Database Module for SONARES.

This module handles the storage and management of material properties
relevant to acoustic simulations.
"""

import pandas as pd
import numpy as np

# Default material database with acoustic properties
DEFAULT_MATERIALS = {
    "Wood (Pine)": {
        "resonance_frequency": 440,  # Hz
        "density": 500,  # kg/m^3
        "damping_coefficient": 0.05,
        "elasticity": 0.8,
        "destruction_threshold": 140,  # dB SPL (approximate)
        "color": "#8B4513"
    },
    "Concrete": {
        "resonance_frequency": 120,  # Hz
        "density": 2400,  # kg/m^3
        "damping_coefficient": 0.02,
        "elasticity": 0.2,
        "destruction_threshold": 180,  # dB SPL (approximate)
        "color": "#808080"
    },
    "Clay Brick": {
        "resonance_frequency": 270,  # Hz
        "density": 1800,  # kg/m^3
        "damping_coefficient": 0.03,
        "elasticity": 0.4,
        "destruction_threshold": 160,  # dB SPL (approximate)
        "color": "#CD5C5C"
    },
    "Steel": {
        "resonance_frequency": 1200,  # Hz
        "density": 7800,  # kg/m^3
        "damping_coefficient": 0.001,
        "elasticity": 0.1,
        "destruction_threshold": 200,  # dB SPL (approximate)
        "color": "#708090"
    },
    "Glass": {
        "resonance_frequency": 800,  # Hz
        "density": 2500,  # kg/m^3
        "damping_coefficient": 0.01,
        "elasticity": 0.3,
        "destruction_threshold": 130,  # dB SPL (approximate)
        "color": "#ADD8E6"
    },
    "Plaster": {
        "resonance_frequency": 320,  # Hz
        "density": 1000,  # kg/m^3
        "damping_coefficient": 0.08,
        "elasticity": 0.6,
        "destruction_threshold": 120,  # dB SPL (approximate)
        "color": "#F5F5F5"
    }
}

class MaterialDatabase:
    """Class to manage material properties for acoustic simulations."""
    
    def __init__(self):
        """Initialize material database with default values."""
        self.materials = DEFAULT_MATERIALS.copy()
        self.df = self._create_dataframe()
        
    def _create_dataframe(self):
        """Convert materials dictionary to pandas DataFrame."""
        return pd.DataFrame.from_dict(self.materials, orient='index')
    
    def get_material_names(self):
        """Get a list of all available material names."""
        return list(self.materials.keys())
    
    def get_material_properties(self, material_name):
        """
        Get properties for a specific material.
        
        Args:
            material_name (str): Name of the material
            
        Returns:
            dict: Material properties or None if not found
        """
        return self.materials.get(material_name, None)
    
    def add_custom_material(self, name, properties):
        """
        Add a new custom material to the database.
        
        Args:
            name (str): Name of the new material
            properties (dict): Material properties including resonance_frequency, 
                               density, damping_coefficient, elasticity, etc.
        """
        self.materials[name] = properties
        self.df = self._create_dataframe()
        
    def calculate_resonance_response(self, material_name, frequency):
        """
        Calculate the resonance response of a material to a given frequency.
        
        Args:
            material_name (str): Name of the material
            frequency (float): Applied frequency in Hz
            
        Returns:
            float: Resonance response factor (0-1)
        """
        material = self.get_material_properties(material_name)
        if not material:
            return 0.0
        
        resonance_freq = material["resonance_frequency"]
        damping = material["damping_coefficient"]
        
        # Simple resonance response model - can be refined with more complex physics
        f_ratio = frequency / resonance_freq
        response = 1.0 / np.sqrt((1 - f_ratio**2)**2 + (2 * damping * f_ratio)**2)
        
        # Normalize response to 0-1 range
        return min(1.0, response / 10.0)
    
    def get_material_dataframe(self):
        """Get materials as a pandas DataFrame for display and analysis."""
        return self.df
