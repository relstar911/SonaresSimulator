"""
Batch Testing Module for SONARES.

This module provides functionality for running multiple simulations
and comparing them with experimental data in batches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

class BatchTestingSystem:
    """Class to handle batch simulation testing and experimental data comparison."""
    
    def __init__(self, simulation_engine, database_handler=None):
        """
        Initialize the batch testing system.
        
        Args:
            simulation_engine: Instance of AcousticSimulation class
            database_handler: Optional database handler for storing results
        """
        self.simulation_engine = simulation_engine
        self.database_handler = database_handler
        self.results = {}
        self.experimental_data = {}
        
    def load_experimental_data(self):
        """
        Load all available experimental data files.
        
        Returns:
            dict: Dictionary with material names as keys and dataframes as values
        """
        data_files = glob.glob("experimental_data/*.csv")
        experimental_data = {}
        
        for file_path in data_files:
            try:
                # Extract material name from filename
                filename = os.path.basename(file_path)
                # Remove extension and replace underscores with spaces
                material_name = filename.replace(".csv", "").replace("_", " ")
                
                # Load the CSV data
                data = pd.read_csv(file_path)
                experimental_data[material_name] = data
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        self.experimental_data = experimental_data
        return experimental_data
    
    def run_batch_test(self, materials=None, frequencies=None, parameters=None):
        """
        Run a batch of simulations with different parameters.
        
        Args:
            materials (list): List of material names to test, or None for all materials
            frequencies (list): List of frequencies to test, or None for default range
            parameters (dict): Additional simulation parameters
            
        Returns:
            dict: Results for each material and frequency combination
        """
        # Default parameters
        if parameters is None:
            parameters = {
                'medium': 'air',
                'reflection_coefficient': 0.0
            }
        
        # Use all available materials if none specified
        if materials is None:
            materials = self.simulation_engine.material_db.get_material_names()
        
        # Use default frequency range if none specified
        if frequencies is None:
            frequencies = np.linspace(50, 1000, 20)  # 20 points from 50Hz to 1000Hz
        
        # Store results for each material and frequency
        results = {}
        
        # Run simulations for each material
        for material in materials:
            material_results = {}
            
            # Set the material in simulation engine
            self.simulation_engine.set_material(material)
            
            # Run simulation for each frequency
            for freq in frequencies:
                # Set frequency
                self.simulation_engine.set_frequency(freq)
                
                # Run simulation with the given parameters
                sim_result = self.simulation_engine.run_simulation(
                    medium=parameters.get('medium', 'air'),
                    reflection_coefficient=parameters.get('reflection_coefficient', 0.0)
                )
                
                # Skip if simulation failed
                if "error" in sim_result:
                    print(f"Error in simulation for {material} at {freq}Hz: {sim_result['error']}")
                    continue
                
                # Store relevant results
                material_results[freq] = {
                    'max_intensity': sim_result.get('max_intensity', 0),
                    'resonance_factor': sim_result.get('resonance_factor', 0),
                    'resonance_hotspots': sim_result.get('hotspot_count', 0)
                }
            
            # Store all results for this material
            results[material] = material_results
        
        self.results = results
        return results
    
    def compare_with_experimental(self, material):
        """
        Compare simulation results with experimental data for a specific material.
        
        Args:
            material (str): Material name to compare
            
        Returns:
            dict: Comparison metrics and data
        """
        # Check if we have simulation results and experimental data for this material
        if material not in self.results:
            return {"error": f"No simulation results found for {material}"}
        
        # Find matching experimental data
        exp_data = None
        for exp_material in self.experimental_data:
            if material.lower() in exp_material.lower():
                exp_data = self.experimental_data[exp_material]
                break
                
        if exp_data is None:
            return {"error": f"No experimental data found for {material}"}
        
        # Extract simulation frequencies and intensities
        sim_freqs = sorted(list(self.results[material].keys()))
        sim_intensities = [self.results[material][freq]['max_intensity'] for freq in sim_freqs]
        
        # Extract experimental frequencies and intensities
        if 'Frequency (Hz)' in exp_data and 'Measured Intensity' in exp_data:
            exp_freqs = exp_data['Frequency (Hz)'].values
            exp_intensities = exp_data['Measured Intensity'].values
            
            # Interpolate simulation data to match experimental frequencies
            sim_interp = interp1d(sim_freqs, sim_intensities, bounds_error=False, fill_value='extrapolate')
            sim_at_exp_freq = sim_interp(exp_freqs)
            
            # Calculate metrics
            correlation, p_value = pearsonr(sim_at_exp_freq, exp_intensities)
            mae = np.mean(np.abs(sim_at_exp_freq - exp_intensities))
            mse = np.mean((sim_at_exp_freq - exp_intensities)**2)
            
            return {
                "correlation": correlation,
                "p_value": p_value,
                "mae": mae,
                "mse": mse,
                "simulation_data": {
                    "frequencies": sim_freqs,
                    "intensities": sim_intensities
                },
                "experimental_data": {
                    "frequencies": exp_freqs.tolist(),
                    "intensities": exp_intensities.tolist()
                },
                "interpolated_simulation": {
                    "frequencies": exp_freqs.tolist(),
                    "intensities": sim_at_exp_freq.tolist()
                }
            }
        else:
            return {"error": "Experimental data does not contain required columns"}
    
    def compare_all_materials(self):
        """
        Compare simulation results with experimental data for all materials.
        
        Returns:
            dict: Comparison metrics for each material
        """
        comparison_results = {}
        
        for material in self.results:
            comparison = self.compare_with_experimental(material)
            if "error" not in comparison:
                comparison_results[material] = comparison
        
        return comparison_results
    
    def plot_comparison(self, material, figsize=(12, 6)):
        """
        Create a plot comparing simulation and experimental results for a material.
        
        Args:
            material (str): Material name to plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.Figure: The generated figure
        """
        comparison = self.compare_with_experimental(material)
        
        if "error" in comparison:
            print(f"Error: {comparison['error']}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot simulation data
        sim_freqs = comparison["simulation_data"]["frequencies"]
        sim_intensities = comparison["simulation_data"]["intensities"]
        ax.plot(sim_freqs, sim_intensities, 'b-', label="Simulation")
        
        # Plot experimental data
        exp_freqs = comparison["experimental_data"]["frequencies"]
        exp_intensities = comparison["experimental_data"]["intensities"]
        ax.scatter(exp_freqs, exp_intensities, color='red', s=50, alpha=0.7, label="Experimental")
        
        # Add labels and legend
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Simulation vs. Experimental Data: {material}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add metrics as text
        metrics_text = (
            f"Correlation: {comparison['correlation']:.3f}\n"
            f"MAE: {comparison['mae']:.3f}\n"
            f"MSE: {comparison['mse']:.3f}"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
        
        return fig
    
    def generate_report(self, save_path=None):
        """
        Generate a comprehensive report of all batch tests and comparisons.
        
        Args:
            save_path (str): Optional path to save the report
            
        Returns:
            pd.DataFrame: Summary of comparison metrics for all materials
        """
        comparisons = self.compare_all_materials()
        
        # Create summary dataframe
        summary_data = []
        for material, comparison in comparisons.items():
            summary_data.append({
                'Material': material,
                'Correlation': comparison['correlation'],
                'P-value': comparison['p_value'],
                'MAE': comparison['mae'],
                'MSE': comparison['mse']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by correlation (best match first)
        summary_df = summary_df.sort_values('Correlation', ascending=False)
        
        # Save if path provided
        if save_path:
            summary_df.to_csv(save_path, index=False)
            
        return summary_df
    
    def optimize_parameters(self, material, param_ranges, metric='correlation'):
        """
        Find optimal simulation parameters to match experimental data.
        
        Args:
            material (str): Material name to optimize for
            param_ranges (dict): Dictionary of parameter ranges to search
                (e.g., {'reflection_coefficient': [0.0, 0.1, 0.2, ..., 0.9]})
            metric (str): Metric to optimize ('correlation', 'mae', or 'mse')
            
        Returns:
            dict: Optimal parameters and resulting metrics
        """
        if material not in self.simulation_engine.material_db.get_material_names():
            return {"error": f"Material {material} not found"}
            
        # Find matching experimental data
        exp_data = None
        for exp_material in self.experimental_data:
            if material.lower() in exp_material.lower():
                exp_data = self.experimental_data[exp_material]
                break
                
        if exp_data is None:
            return {"error": f"No experimental data found for {material}"}
        
        # Extract experimental frequencies
        if 'Frequency (Hz)' not in exp_data or 'Measured Intensity' not in exp_data:
            return {"error": "Experimental data does not contain required columns"}
            
        exp_freqs = exp_data['Frequency (Hz)'].values
        
        # Generate all parameter combinations
        import itertools
        param_keys = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        best_value = -float('inf') if metric == 'correlation' else float('inf')
        best_params = None
        best_metrics = None
        
        # Test each parameter combination
        for param_combination in itertools.product(*param_values):
            # Create parameter dictionary
            params = dict(zip(param_keys, param_combination))
            
            # Run simulation for all experimental frequencies
            self.simulation_engine.set_material(material)
            sim_results = {}
            
            for freq in exp_freqs:
                self.simulation_engine.set_frequency(freq)
                result = self.simulation_engine.run_simulation(**params)
                
                if "error" in result:
                    continue
                    
                sim_results[freq] = result.get('max_intensity', 0)
            
            # Skip if any simulation failed
            if len(sim_results) != len(exp_freqs):
                continue
                
            # Calculate metrics
            sim_intensities = [sim_results[freq] for freq in exp_freqs]
            exp_intensities = exp_data['Measured Intensity'].values
            
            correlation, _ = pearsonr(sim_intensities, exp_intensities)
            mae = np.mean(np.abs(np.array(sim_intensities) - exp_intensities))
            mse = np.mean((np.array(sim_intensities) - exp_intensities)**2)
            
            # Check if this is the best result so far
            current_value = correlation if metric == 'correlation' else -mae if metric == 'mae' else -mse
            
            if (metric == 'correlation' and current_value > best_value) or \
               ((metric == 'mae' or metric == 'mse') and current_value > best_value):
                best_value = current_value
                best_params = params
                best_metrics = {
                    'correlation': correlation,
                    'mae': mae,
                    'mse': mse
                }
        
        if best_params is None:
            return {"error": "Optimization failed. No valid parameter combinations found."}
            
        return {
            "optimal_parameters": best_params,
            "metrics": best_metrics
        }