"""
Visualization Module for SONARES.

This module provides visualization functions for the acoustic simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap for acoustic visualization
ACOUSTIC_CMAP = LinearSegmentedColormap.from_list(
    'acoustic', 
    [(0, 'black'), (0.3, 'blue'), (0.6, 'yellow'), (0.8, 'orange'), (1, 'red')]
)

def create_2d_heatmap(slice_data, title="Acoustic Intensity"):
    """
    Create a 2D heatmap visualization from a slice of the simulation.
    
    Args:
        slice_data (dict): Slice data from simulation
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if slice_data is None:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No simulation data available", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Extract data from slice
    x_grid = slice_data['x_grid']
    y_grid = slice_data['y_grid']
    field = slice_data['resonance']  # Use resonance field by default
    axes_labels = slice_data['axes_labels']
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate heatmap
    c = ax.pcolormesh(x_grid, y_grid, field, cmap=ACOUSTIC_CMAP, shading='auto')
    
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Intensity')
    
    # Set labels and title
    ax.set_xlabel(f"{axes_labels[0]} (m)")
    ax.set_ylabel(f"{axes_labels[1]} (m)")
    ax.set_title(title)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def create_3d_visualization(simulation, threshold=0.5):
    """
    Create a 3D visualization of the acoustic field.
    
    Args:
        simulation: Acoustic simulation object with computed fields
        threshold (float): Isosurface threshold value (0-1)
        
    Returns:
        plotly.graph_objects.Figure: The 3D plot figure
    """
    if simulation.resonance_field is None:
        # Create empty 3D plot with message
        fig = go.Figure()
        fig.add_annotation(
            text="No simulation data available",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5], title="X (m)"),
                yaxis=dict(range=[-1.5, 1.5], title="Y (m)"),
                zaxis=dict(range=[-1.5, 1.5], title="Z (m)"),
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            scene_aspectmode='cube',
            title="3D Acoustic Field"
        )
        return fig
    
    # Extract data for visualization
    X, Y, Z = simulation.grid['X'], simulation.grid['Y'], simulation.grid['Z']
    field = simulation.resonance_field
    
    # Create figure
    fig = go.Figure()
    
    # Generate isosurface for the resonance field
    fig.add_trace(go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=field.flatten(),
        isomin=threshold,
        isomax=1.0,
        opacity=0.7,
        surface_count=3,
        colorscale=[
            [0, 'blue'],
            [0.5, 'yellow'],
            [0.8, 'orange'],
            [1, 'red']
        ],
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    # Add source points
    source_positions = simulation.source_config.get_source_positions()
    if len(source_positions) > 0:
        fig.add_trace(go.Scatter3d(
            x=source_positions[:, 0],
            y=source_positions[:, 1],
            z=source_positions[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='white',
                symbol='circle',
                line=dict(color='black', width=1)
            ),
            name='Sound Sources'
        ))
    
    # Add coordinate system origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(
            size=6,
            color='black',
            symbol='circle'
        ),
        name='Origin'
    ))
    
    # Configure the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], title="X (m)"),
            yaxis=dict(range=[-1.5, 1.5], title="Y (m)"),
            zaxis=dict(range=[-1.5, 1.5], title="Z (m)"),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_aspectmode='cube',
        title=f"3D Acoustic Field (Iso-threshold: {threshold:.2f})"
    )
    
    return fig

def plot_frequency_response(freq_data, material_name):
    """
    Create a plot showing frequency response for a material.
    
    Args:
        freq_data (dict): Frequency response data
        material_name (str): Name of the material
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if freq_data is None:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No material selected", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Extract data
    frequencies = freq_data['frequencies']
    responses = freq_data['responses']
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot frequency response
    ax.plot(frequencies, responses, linewidth=2, color='blue')
    
    # Add resonance peak marker
    max_idx = np.argmax(responses)
    ax.plot(frequencies[max_idx], responses[max_idx], 'ro', markersize=8)
    ax.annotate(f'Peak: {frequencies[max_idx]:.1f} Hz', 
                xy=(frequencies[max_idx], responses[max_idx]),
                xytext=(10, -10), textcoords='offset points',
                fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Response Amplitude')
    ax.set_title(f'Frequency Response for {material_name}')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1 * np.max(responses))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_hotspot_analysis(hotspot_data, material_name):
    """
    Create a plot visualizing hotspot analysis results.
    
    Args:
        hotspot_data (dict): Hotspot analysis data
        material_name (str): Name of the material
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if hotspot_data is None or hotspot_data['count'] == 0:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No resonance hotspots detected", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Create figure with 3D projection
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract hotspot data
    positions = hotspot_data['positions']
    intensities = hotspot_data['intensities']
    
    # Normalize intensities for color mapping
    norm_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
    
    # Plot hotspots
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=intensities, cmap='hot', s=50*norm_intensities+10, alpha=0.7
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Intensity')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Resonance Hotspots in {material_name}\n({hotspot_data["count"]} points, max intensity: {hotspot_data["max_intensity"]:.3f})')
    
    # Set axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    return fig
