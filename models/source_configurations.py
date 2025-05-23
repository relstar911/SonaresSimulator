"""
Source Configurations Module for SONARES.

This module provides functionality for configuring acoustic sources
in the virtual 3D space.
"""

import numpy as np
from enum import Enum

class SourceArrangement(Enum):
    """Enumeration of available source arrangement patterns."""
    LINEAR = "Linear"
    CIRCULAR = "Circular"
    SPHERICAL = "Spherical"
    GRID = "3D Grid"
    PLANAR = "2D Plane"
    CUSTOM = "Custom Points"

class SourceConfiguration:
    """Class to manage acoustic source configurations and placements."""
    
    def __init__(self, room_size=(3, 3, 3)):
        """
        Initialize the source configuration.
        
        Args:
            room_size (tuple): Dimensions of the room in meters (x, y, z)
        """
        self.room_size = np.array(room_size)
        self.sources = []
        self.source_count = 0
        
    def generate_sources(self, arrangement, count, params=None):
        """
        Generate sources according to the specified arrangement.
        
        Args:
            arrangement (SourceArrangement): Type of arrangement
            count (int): Number of sources to generate
            params (dict): Additional parameters for the arrangement
                
        Returns:
            list: Generated source positions as (x, y, z) tuples
        """
        self.source_count = count
        self.sources = []
        
        if params is None:
            params = {}
            
        if arrangement == SourceArrangement.LINEAR:
            self._generate_linear_arrangement(count, params)
        elif arrangement == SourceArrangement.CIRCULAR:
            self._generate_circular_arrangement(count, params)
        elif arrangement == SourceArrangement.SPHERICAL:
            self._generate_spherical_arrangement(count, params)
        elif arrangement == SourceArrangement.GRID:
            self._generate_grid_arrangement(count, params)
        elif arrangement == SourceArrangement.PLANAR:
            self._generate_planar_arrangement(count, params)
        elif arrangement == SourceArrangement.CUSTOM:
            # Custom arrangement handled separately through add_source method
            pass
            
        return self.sources
    
    def _generate_linear_arrangement(self, count, params):
        """
        Generate sources in a linear arrangement.
        
        Args:
            count (int): Number of sources
            params (dict): Parameters including start_point, end_point
        """
        start_point = np.array(params.get('start_point', [-1, 0, 0]))
        end_point = np.array(params.get('end_point', [1, 0, 0]))
        
        for i in range(count):
            t = i / max(1, count - 1)
            position = start_point + t * (end_point - start_point)
            self.add_source(position, i)
    
    def _generate_circular_arrangement(self, count, params):
        """
        Generate sources in a circular arrangement.
        
        Args:
            count (int): Number of sources
            params (dict): Parameters including center, radius, and axis
        """
        center = np.array(params.get('center', [0, 0, 0]))
        radius = params.get('radius', 1.0)
        axis = params.get('axis', 'z')  # Axis perpendicular to circle plane
        
        for i in range(count):
            angle = 2 * np.pi * i / count
            if axis == 'x':
                position = center + np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
            elif axis == 'y':
                position = center + np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
            else:  # axis == 'z'
                position = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            
            self.add_source(position, i)
    
    def _generate_spherical_arrangement(self, count, params):
        """
        Generate sources in a spherical arrangement using the Fibonacci sphere algorithm.
        
        Args:
            count (int): Number of sources
            params (dict): Parameters including center and radius
        """
        center = np.array(params.get('center', [0, 0, 0]))
        radius = params.get('radius', 1.0)
        
        # Fibonacci sphere algorithm for roughly uniform point distribution on sphere
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(count):
            y = 1 - (2 * i) / (count - 1 if count > 1 else 1)
            radius_at_y = np.sqrt(1 - y*y)
            
            theta = 2 * np.pi * i / phi
            x = radius_at_y * np.cos(theta)
            z = radius_at_y * np.sin(theta)
            
            position = center + radius * np.array([x, y, z])
            self.add_source(position, i)
    
    def add_source(self, position, source_id=None):
        """
        Add a single source at the specified position.
        
        Args:
            position (list/tuple/ndarray): 3D position [x, y, z]
            source_id (int, optional): Unique ID for the source
        
        Returns:
            int: The ID of the added source
        """
        position = np.array(position)
        
        # Ensure position is within room boundaries
        position = np.clip(position, -self.room_size/2, self.room_size/2)
        
        if source_id is None:
            source_id = len(self.sources)
            
        source = {
            'id': source_id,
            'position': position,
            'frequency': 440.0,  # Default frequency (Hz)
            'phase': 0.0,        # Default phase (degrees)
            'amplitude': 1.0,    # Default amplitude
            'active': True       # Default state
        }
        
        # Replace existing source if ID already exists
        for i, s in enumerate(self.sources):
            if s['id'] == source_id:
                self.sources[i] = source
                return source_id
                
        self.sources.append(source)
        return source_id
    
    def update_source(self, source_id, **kwargs):
        """
        Update properties of an existing source.
        
        Args:
            source_id (int): ID of the source to update
            **kwargs: Source properties to update (position, frequency, phase, etc.)
            
        Returns:
            bool: True if source was updated, False if not found
        """
        for i, source in enumerate(self.sources):
            if source['id'] == source_id:
                self.sources[i].update(kwargs)
                return True
        return False
    
    def get_source_positions(self):
        """
        Get positions of all active sources.
        
        Returns:
            ndarray: Array of source positions with shape (n, 3)
        """
        active_sources = [s for s in self.sources if s['active']]
        if not active_sources:
            return np.zeros((0, 3))
        return np.array([s['position'] for s in active_sources])
    
    def get_source_properties(self):
        """
        Get properties of all active sources.
        
        Returns:
            dict: Dictionary with arrays of positions, frequencies, phases, and amplitudes
        """
        active_sources = [s for s in self.sources if s['active']]
        
        if not active_sources:
            return {
                'positions': np.zeros((0, 3)),
                'frequencies': np.array([]),
                'phases': np.array([]),
                'amplitudes': np.array([])
            }
            
        return {
            'positions': np.array([s['position'] for s in active_sources]),
            'frequencies': np.array([s['frequency'] for s in active_sources]),
            'phases': np.array([s['phase'] for s in active_sources]),
            'amplitudes': np.array([s['amplitude'] for s in active_sources])
        }
    
    def set_uniform_frequency(self, frequency):
        """
        Set the same frequency for all sources.
        
        Args:
            frequency (float): Frequency in Hz
        """
        for source in self.sources:
            source['frequency'] = frequency
    
    def set_uniform_phase(self, phase):
        """
        Set the same phase for all sources.
        
        Args:
            phase (float): Phase in degrees (0-360)
        """
        for source in self.sources:
            source['phase'] = phase
    
    def set_uniform_amplitude(self, amplitude):
        """
        Set the same amplitude for all sources.
        
        Args:
            amplitude (float): Amplitude (0-1)
        """
        for source in self.sources:
            source['amplitude'] = amplitude
            
    def _generate_grid_arrangement(self, count, params):
        """
        Generate sources in a 3D grid arrangement.
        
        Args:
            count (int): Number of sources
            params (dict): Parameters including dimensions and spacing
        """
        # Get grid dimensions (trying to keep the total close to count)
        total_points = count
        
        # Calculate dimensions for a roughly cubic grid
        dim = round(total_points ** (1/3))
        nx = dim
        ny = dim
        nz = max(1, round(total_points / (nx * ny)))
        
        # Adjust to match requested count as closely as possible
        while nx * ny * nz > count and nx > 1:
            nx -= 1
        while nx * ny * nz > count and ny > 1:
            ny -= 1
        while nx * ny * nz > count and nz > 1:
            nz -= 1
            
        # Center of the grid
        center = np.array(params.get('center', [0, 0, 0]))
        
        # Spacing between points
        spacing = params.get('spacing', 0.5)
        
        # Size of the grid
        size_x = (nx - 1) * spacing
        size_y = (ny - 1) * spacing
        size_z = (nz - 1) * spacing
        
        # Starting position (bottom left corner)
        start = center - np.array([size_x / 2, size_y / 2, size_z / 2])
        
        # Create the grid
        source_id = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if source_id < count:
                        position = start + np.array([i * spacing, j * spacing, k * spacing])
                        self.add_source(position, source_id)
                        source_id += 1
    
    def _generate_planar_arrangement(self, count, params):
        """
        Generate sources in a 2D planar arrangement.
        
        Args:
            count (int): Number of sources
            params (dict): Parameters including plane, center and dimensions
        """
        # Which plane to use (xy, xz, yz)
        plane = params.get('plane', 'xy')
        
        # Calculate dimensions for a roughly square grid
        dim = round(count ** 0.5)
        nx = dim
        ny = dim
        
        # Adjust to match requested count as closely as possible
        while nx * ny > count and nx > 1:
            nx -= 1
        
        # Center of the plane
        center = np.array(params.get('center', [0, 0, 0]))
        
        # Spacing between points
        spacing = params.get('spacing', 0.5)
        
        # Size of the plane
        size_1 = (nx - 1) * spacing
        size_2 = (ny - 1) * spacing
        
        # Create the grid based on the plane
        source_id = 0
        
        if plane == 'xy':
            # Starting position (bottom left corner)
            start = center - np.array([size_1 / 2, size_2 / 2, 0])
            
            for i in range(nx):
                for j in range(ny):
                    if source_id < count:
                        position = start + np.array([i * spacing, j * spacing, 0])
                        self.add_source(position, source_id)
                        source_id += 1
                        
        elif plane == 'xz':
            # Starting position (bottom left corner)
            start = center - np.array([size_1 / 2, 0, size_2 / 2])
            
            for i in range(nx):
                for j in range(ny):
                    if source_id < count:
                        position = start + np.array([i * spacing, 0, j * spacing])
                        self.add_source(position, source_id)
                        source_id += 1
                        
        elif plane == 'yz':
            # Starting position (bottom left corner)
            start = center - np.array([0, size_1 / 2, size_2 / 2])
            
            for i in range(nx):
                for j in range(ny):
                    if source_id < count:
                        position = start + np.array([0, i * spacing, j * spacing])
                        self.add_source(position, source_id)
                        source_id += 1
