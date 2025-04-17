"""
Acoustic Physics Module for SONARES.

This module provides functions for acoustic wave simulation
and physics calculations.
"""

import numpy as np

def calculate_wave_interference(grid, source_positions, frequencies, phases, amplitudes):
    """
    Calculate the acoustic wave interference pattern from multiple sources.
    
    Args:
        grid (dict): Dictionary containing X, Y, Z meshgrids
        source_positions (ndarray): Array of source positions with shape (n, 3)
        frequencies (ndarray): Array of source frequencies (Hz)
        phases (ndarray): Array of source phases (degrees)
        amplitudes (ndarray): Array of source amplitudes
        
    Returns:
        ndarray: 3D array representing the interference field
    """
    # Speed of sound in air (m/s)
    speed_of_sound = 343.0
    
    # Convert phases to radians
    phases_rad = np.radians(phases)
    
    # Initialize result array
    X, Y, Z = grid['X'], grid['Y'], grid['Z']
    result = np.zeros_like(X, dtype=np.float64)
    
    # Convert all to numpy arrays for vectorized operations
    source_positions = np.array(source_positions)
    frequencies = np.array(frequencies)
    phases_rad = np.array(phases_rad)
    amplitudes = np.array(amplitudes)
    
    # For each source, calculate the wave field and add to result
    for i in range(len(source_positions)):
        pos = source_positions[i]
        freq = frequencies[i]
        phase = phases_rad[i]
        amplitude = amplitudes[i]
        
        # Calculate distance from each point to the source
        distances = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)
        
        # Calculate wavelength
        wavelength = speed_of_sound / freq
        
        # Calculate wave number
        k = 2 * np.pi / wavelength
        
        # Calculate the wave amplitude at each point (1/r attenuation)
        wave = amplitude * np.sin(k * distances + phase) / distances
        
        # Add to the result
        result += wave
    
    # Normalize the result to be between 0 and 1
    if np.max(np.abs(result)) > 0:
        result = (result - np.min(result)) / (np.max(result) - np.min(result))
    
    return result

def calculate_energy_density(interference_field):
    """
    Calculate energy density from interference field.
    
    Args:
        interference_field (ndarray): 3D array of interference pattern
        
    Returns:
        ndarray: 3D array of energy density
    """
    # In this simplified model, energy density is proportional to squared amplitude
    return interference_field**2

def calculate_material_response(interference_field, material_properties, frequency):
    """
    Calculate the response of a material to an acoustic field.
    
    Args:
        interference_field (ndarray): 3D array of interference pattern
        material_properties (dict): Material properties
        frequency (float): Operating frequency in Hz
        
    Returns:
        ndarray: 3D array of material response
    """
    # Extract relevant material properties
    resonance_freq = material_properties["resonance_frequency"]
    damping = material_properties["damping_coefficient"]
    
    # Calculate frequency ratio
    f_ratio = frequency / resonance_freq
    
    # Resonance amplitude factor using damped oscillator model
    resonance_factor = 1.0 / np.sqrt((1 - f_ratio**2)**2 + (2 * damping * f_ratio)**2)
    
    # Cap the resonance factor to avoid unrealistic values
    resonance_factor = min(10.0, resonance_factor)
    
    # Scale the interference field by the resonance factor
    return interference_field * resonance_factor

def calculate_wavelength(frequency, medium="air"):
    """
    Calculate wavelength for a given frequency.
    
    Args:
        frequency (float): Frequency in Hz
        medium (str): Propagation medium
        
    Returns:
        float: Wavelength in meters
    """
    # Speed of sound in different media (m/s)
    speeds = {
        "air": 343.0,
        "water": 1481.0,
        "wood": 3800.0,
        "concrete": 3200.0,
        "steel": 5100.0
    }
    
    c = speeds.get(medium.lower(), 343.0)  # Default to air if medium not recognized
    return c / frequency
