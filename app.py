"""
SONARES - Sonic Resonance Evaluation System

A scientific simulation tool for acoustic resonance analysis of materials 
in a controlled virtual environment.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from datetime import datetime

from models.acoustic_simulation import AcousticSimulation
from models.source_configurations import SourceArrangement
from utils.visualization import (
    create_2d_heatmap,
    create_3d_visualization,
    plot_frequency_response,
    plot_hotspot_analysis
)

# Check if database functionality is available
try:
    from models.database import DatabaseHandler
    db_available = True
    db_handler = DatabaseHandler()
except ImportError as e:
    db_available = False
    print(f"Database import error: {e}")
except Exception as e:
    db_available = False
    print(f"Database connection error: {e}")

# Set page configuration
st.set_page_config(
    page_title="SONARES - Sonic Resonance Evaluation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulation' not in st.session_state:
    st.session_state.simulation = AcousticSimulation()
    
if 'selected_material' not in st.session_state:
    st.session_state.selected_material = None
    
if 'selected_frequency' not in st.session_state:
    st.session_state.selected_frequency = 440.0  # Default to A4 (440 Hz)
    
if 'source_arrangement' not in st.session_state:
    st.session_state.source_arrangement = SourceArrangement.CIRCULAR
    
if 'source_count' not in st.session_state:
    st.session_state.source_count = 8
    
if 'needs_update' not in st.session_state:
    st.session_state.needs_update = True
    
if 'slice_axis' not in st.session_state:
    st.session_state.slice_axis = 'z'
    
if 'slice_position' not in st.session_state:
    st.session_state.slice_position = 0.0

# Main title
st.title("SONARES - Sonic Resonance Evaluation System")
st.markdown("""
A scientific simulation tool for acoustic resonance analysis of materials 
in a controlled virtual environment.
""")

# Create page layout with sidebar and main content
with st.sidebar:
    st.header("Simulation Controls")
    
    # Material selection
    st.subheader("Material")
    material_names = st.session_state.simulation.material_db.get_material_names()
    selected_material = st.selectbox(
        "Select material",
        material_names,
        index=0
    )
    
    if selected_material != st.session_state.selected_material:
        st.session_state.selected_material = selected_material
        st.session_state.simulation.set_material(selected_material)
        st.session_state.needs_update = True
    
    # Frequency control
    st.subheader("Acoustic Parameters")
    freq_min, freq_max = 20, 2000
    selected_frequency = st.slider(
        "Frequency (Hz)",
        min_value=freq_min,
        max_value=freq_max,
        value=int(st.session_state.selected_frequency),
        step=5
    )
    
    if selected_frequency != st.session_state.selected_frequency:
        st.session_state.selected_frequency = selected_frequency
        st.session_state.simulation.set_frequency(selected_frequency)
        st.session_state.needs_update = True
    
    # Source configuration
    st.subheader("Source Configuration")
    source_arrangement = st.selectbox(
        "Arrangement",
        [arrangement.value for arrangement in SourceArrangement],
        index=[arr.value for arr in SourceArrangement].index(st.session_state.source_arrangement.value)
    )
    
    source_count = st.slider(
        "Number of sources",
        min_value=1,
        max_value=32,
        value=st.session_state.source_count,
        step=1
    )
    
    # Check if source configuration changed
    arrangement_changed = (
        source_arrangement != st.session_state.source_arrangement.value or
        source_count != st.session_state.source_count
    )
    
    if arrangement_changed:
        st.session_state.source_arrangement = next(
            arr for arr in SourceArrangement if arr.value == source_arrangement
        )
        st.session_state.source_count = source_count
        
        # Generate new source configuration
        params = {}
        if st.session_state.source_arrangement == SourceArrangement.LINEAR:
            params = {
                'start_point': [-1.2, 0, 0],
                'end_point': [1.2, 0, 0]
            }
        elif st.session_state.source_arrangement == SourceArrangement.CIRCULAR:
            params = {
                'center': [0, 0, 0],
                'radius': 1.2,
                'axis': 'z'
            }
        elif st.session_state.source_arrangement == SourceArrangement.SPHERICAL:
            params = {
                'center': [0, 0, 0],
                'radius': 1.2
            }
            
        st.session_state.simulation.source_config.generate_sources(
            st.session_state.source_arrangement,
            st.session_state.source_count,
            params
        )
        st.session_state.needs_update = True
    
    # Advanced physics controls
    st.subheader("Physics Settings")
    
    # Propagation medium
    if 'medium' not in st.session_state:
        st.session_state.medium = "air"
        
    medium = st.selectbox(
        "Propagation medium",
        ["air", "water", "wood", "concrete", "steel"],
        index=["air", "water", "wood", "concrete", "steel"].index(st.session_state.medium)
    )
    
    if medium != st.session_state.medium:
        st.session_state.medium = medium
        st.session_state.needs_update = True
    
    # Room reflection coefficient
    if 'reflection_coefficient' not in st.session_state:
        st.session_state.reflection_coefficient = 0.0
        
    reflection = st.slider(
        "Wall reflection",
        min_value=0.0,
        max_value=0.9,
        value=float(st.session_state.reflection_coefficient),
        step=0.1,
        help="Higher values increase sound reflections from walls (0 = anechoic, 0.9 = highly reflective)"
    )
    
    if reflection != st.session_state.reflection_coefficient:
        st.session_state.reflection_coefficient = reflection
        st.session_state.needs_update = True
    
    # Visualization controls
    st.subheader("Visualization")
    slice_axis = st.selectbox(
        "Slice axis",
        ['x', 'y', 'z'],
        index=['x', 'y', 'z'].index(st.session_state.slice_axis)
    )
    
    slice_position = st.slider(
        "Slice position (m)",
        min_value=-1.5,
        max_value=1.5,
        value=float(st.session_state.slice_position),
        step=0.1
    )
    
    if (slice_axis != st.session_state.slice_axis or 
        slice_position != st.session_state.slice_position):
        st.session_state.slice_axis = slice_axis
        st.session_state.slice_position = slice_position
        st.session_state.needs_update = True
    
    # Run simulation button
    if st.button("Run Simulation"):
        st.session_state.needs_update = True
        
    # Show current speed of sound
    speeds = {
        "air": 343.0,
        "water": 1481.0,
        "wood": 3800.0,
        "concrete": 3200.0,
        "steel": 5100.0
    }
    st.info(f"Speed of sound in {medium}: {speeds.get(medium, 343.0)} m/s")
    
    # Additional information
    st.markdown("---")
    st.caption("SONARES - Sonic Resonance Evaluation System")
    st.caption("Virtual Acoustic Simulation Environment (3m x 3m x 3m)")

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "2D Visualization", 
    "3D Visualization", 
    "Material Analysis",
    "Source Configuration",
    "Experimental Data",
    "Database Records"
])

# If simulation needs update, run it
if st.session_state.needs_update:
    with st.spinner("Running simulation with advanced physics..."):
        if st.session_state.selected_material:
            # Run simulation with advanced physics parameters
            sim_results = st.session_state.simulation.run_simulation(
                medium=st.session_state.medium,
                reflection_coefficient=st.session_state.reflection_coefficient
            )
            if "error" in sim_results:
                st.error(sim_results["error"])
            st.session_state.needs_update = False
        else:
            st.warning("Please select a material")

# Tab 1: 2D Visualization
with tab1:
    st.header("2D Acoustic Field Visualization")
    
    # Get 2D slice for visualization
    slice_data = st.session_state.simulation.get_2d_slice(
        axis=st.session_state.slice_axis,
        position=st.session_state.slice_position
    )
    
    # Create heatmap
    if slice_data is not None:
        heatmap = create_2d_heatmap(
            slice_data, 
            title=f"Acoustic Field ({st.session_state.slice_axis.upper()}-Slice at {st.session_state.slice_position:.1f}m)"
        )
        st.pyplot(heatmap)
    else:
        st.info("Run the simulation to see the 2D visualization")
    
    # Display explanation
    st.markdown("""
    ### 2D Visualization Details
    
    This view shows a 2D slice of the acoustic field in the virtual room. The colors represent:
    - **Dark blue**: Low acoustic intensity
    - **Yellow/Orange**: Medium acoustic intensity
    - **Red**: High acoustic intensity (potential resonance areas)
    
    Areas of high intensity (red) indicate where the material may experience the strongest resonance effects.
    """)

# Tab 2: 3D Visualization
with tab2:
    st.header("3D Acoustic Field Visualization")
    
    # Add threshold slider for 3D visualization
    threshold = st.slider(
        "Isosurface threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    # Create 3D visualization
    fig_3d = create_3d_visualization(st.session_state.simulation, threshold)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Display explanation
    st.markdown("""
    ### 3D Visualization Details
    
    This interactive 3D view shows:
    - **Isosurfaces**: Regions of equal acoustic intensity
    - **White points**: Sound sources
    - **Color gradient**: Represents intensity (blue to red)
    
    You can:
    - **Rotate**: Click and drag
    - **Zoom**: Scroll wheel
    - **Pan**: Right-click and drag
    """)

# Tab 3: Material Analysis
with tab3:
    st.header("Material Analysis")
    
    material = st.session_state.selected_material
    if material:
        # Material properties
        props = st.session_state.simulation.material_db.get_material_properties(material)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Material Properties")
            
            # Create properties table - convert all values to strings to avoid Arrow issues
            props_df = pd.DataFrame({
                'Property': [
                    'Resonance Frequency', 
                    'Density', 
                    'Damping Coefficient',
                    'Elasticity',
                    'Destruction Threshold'
                ],
                'Value': [
                    f"{props['resonance_frequency']} Hz",
                    f"{props['density']} kg/m³",
                    f"{props['damping_coefficient']}",  # Convert to string
                    f"{props['elasticity']}",           # Convert to string
                    f"{props['destruction_threshold']} dB SPL"
                ]
            })
            
            st.dataframe(props_df, use_container_width=True)
            
            # Frequency response curve
            st.subheader("Frequency Response")
            freq_data = st.session_state.simulation.get_frequency_response()
            freq_fig = plot_frequency_response(freq_data, material)
            st.pyplot(freq_fig)
        
        with col2:
            st.subheader("Resonance Analysis")
            
            # Hotspot analysis
            hotspot_threshold = st.slider(
                "Hotspot detection threshold",
                min_value=0.3,
                max_value=0.9,
                value=0.7,
                step=0.05
            )
            
            # Only run hotspot analysis if simulation has been run
            if st.session_state.simulation.resonance_field is not None:
                hotspot_data = st.session_state.simulation.analyze_hotspots(hotspot_threshold)
                
                # Display hotspot summary
                if hotspot_data["count"] > 0:
                    st.success(f"Detected {hotspot_data['count']} resonance hotspots")
                    st.metric("Maximum Resonance Intensity", f"{hotspot_data['max_intensity']:.3f}")
                    
                    # Plot hotspots
                    hotspot_fig = plot_hotspot_analysis(hotspot_data, material)
                    st.pyplot(hotspot_fig)
                else:
                    st.info("No resonance hotspots detected at the current threshold")
            else:
                st.info("Run the simulation to see resonance analysis")
    else:
        st.info("Select a material to see analysis")

# Tab 4: Source Configuration
with tab4:
    st.header("Acoustic Source Configuration")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Source Positions")
        
        # Get source positions
        source_positions = st.session_state.simulation.source_config.get_source_positions()
        
        if len(source_positions) > 0:
            # Create source position table
            source_df = pd.DataFrame(
                source_positions,
                columns=["X (m)", "Y (m)", "Z (m)"]
            )
            source_df.index.name = "Source #"
            source_df.index = source_df.index + 1  # 1-based indexing
            
            st.dataframe(source_df, use_container_width=True)
            
            # Plot source positions in 2D slices
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Define plane limits
            limits = [-1.5, 1.5]
            
            # XY plane (z = 0)
            axes[0].scatter(source_positions[:, 0], source_positions[:, 1], c='red', s=50)
            axes[0].set_xlabel('X (m)')
            axes[0].set_ylabel('Y (m)')
            axes[0].set_title('XY Plane View')
            axes[0].grid(True)
            axes[0].set_xlim(limits)
            axes[0].set_ylim(limits)
            axes[0].set_aspect('equal')
            
            # XZ plane (y = 0)
            axes[1].scatter(source_positions[:, 0], source_positions[:, 2], c='green', s=50)
            axes[1].set_xlabel('X (m)')
            axes[1].set_ylabel('Z (m)')
            axes[1].set_title('XZ Plane View')
            axes[1].grid(True)
            axes[1].set_xlim(limits)
            axes[1].set_ylim(limits)
            axes[1].set_aspect('equal')
            
            # YZ plane (x = 0)
            axes[2].scatter(source_positions[:, 1], source_positions[:, 2], c='blue', s=50)
            axes[2].set_xlabel('Y (m)')
            axes[2].set_ylabel('Z (m)')
            axes[2].set_title('YZ Plane View')
            axes[2].grid(True)
            axes[2].set_xlim(limits)
            axes[2].set_ylim(limits)
            axes[2].set_aspect('equal')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No sources configured. Add sources in the sidebar.")
    
    with col2:
        st.subheader("Configuration Details")
        
        # Display configuration details
        st.markdown(f"""
        **Source arrangement:** {st.session_state.source_arrangement.value}  
        **Number of sources:** {st.session_state.source_count}  
        **Operating frequency:** {st.session_state.selected_frequency} Hz  
        
        **Wavelength in air:** {343 / st.session_state.selected_frequency:.2f} m
        
        Adjust source configuration in the sidebar to control the acoustic field pattern.
        """)
        
        # Arrangement-specific explanations
        st.subheader("Arrangement Description")
        
        if st.session_state.source_arrangement == SourceArrangement.LINEAR:
            st.markdown("""
            **Linear Arrangement**
            
            Sources are arranged in a straight line along the X-axis. This configuration is 
            useful for creating plane wave patterns and directional sound fields.
            """)
        elif st.session_state.source_arrangement == SourceArrangement.CIRCULAR:
            st.markdown("""
            **Circular Arrangement**
            
            Sources are arranged in a circle on the XY plane. This configuration creates
            convergent wave patterns and is useful for focusing acoustic energy.
            """)
        elif st.session_state.source_arrangement == SourceArrangement.SPHERICAL:
            st.markdown("""
            **Spherical Arrangement**
            
            Sources are distributed on a spherical surface. This creates a more uniform
            3D sound field and allows for omnidirectional focusing.
            """)
        elif st.session_state.source_arrangement == SourceArrangement.CUSTOM:
            st.markdown("""
            **Custom Arrangement**
            
            Custom positioning of sources allows for specialized acoustic field patterns.
            """)

# Tab 5: Experimental Data
with tab5:
    st.header("Experimental Data Analysis")
    
    st.markdown("""
    This tab provides tools for importing, analyzing, and comparing experimental data with simulation results.
    You can upload real-world measurements to validate and refine the simulation model.
    """)
    
    # File uploader for experimental data
    uploaded_file = st.file_uploader("Upload experimental data (CSV)", type=['csv'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Import")
        
        if uploaded_file is not None:
            # Create a session state entry for experimental data if it doesn't exist
            if 'experimental_data' not in st.session_state:
                st.session_state.experimental_data = None
                
            try:
                # Read the CSV file into a pandas DataFrame
                exp_data = pd.read_csv(uploaded_file)
                st.session_state.experimental_data = exp_data
                
                # Display the data
                st.dataframe(exp_data, use_container_width=True)
                
                # Show basic statistics
                st.subheader("Data Statistics")
                st.write(exp_data.describe())
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.info("Upload a CSV file with experimental measurements to analyze and compare with simulation.")
            
            # Show example format
            st.subheader("Example Data Format")
            example_data = pd.DataFrame({
                'Frequency (Hz)': [100, 200, 300, 400, 500],
                'Measured Intensity': [0.2, 0.5, 0.8, 0.6, 0.3],
                'Position X (m)': [0, 0, 0, 0, 0],
                'Position Y (m)': [0, 0, 0, 0, 0],
                'Position Z (m)': [0, 0, 0, 0, 0]
            })
            st.dataframe(example_data, use_container_width=True)
    
    with col2:
        st.subheader("Comparative Analysis")
        
        if 'experimental_data' in st.session_state and st.session_state.experimental_data is not None:
            # Plot comparison between experimental and simulation data
            st.markdown("### Simulation vs. Experiment")
            
            # Create a placeholder for the comparison plot
            if st.session_state.simulation.resonance_field is not None and 'Frequency (Hz)' in st.session_state.experimental_data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get the frequency response data from simulation
                freq_data = st.session_state.simulation.get_frequency_response()
                
                if freq_data is not None:
                    # Plot simulation data
                    ax.plot(freq_data['frequencies'], freq_data['responses'], 
                           label='Simulation', color='blue', linewidth=2)
                    
                    # Plot experimental data (assuming it contains frequency and intensity)
                    exp_df = st.session_state.experimental_data
                    if 'Measured Intensity' in exp_df.columns:
                        ax.scatter(exp_df['Frequency (Hz)'], exp_df['Measured Intensity'], 
                                  color='red', s=50, alpha=0.7, label='Experimental')
                    
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Response Intensity')
                    ax.set_title('Simulation vs. Experimental Data')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                    
                    # Calculate correlation between simulation and experiment
                    if 'Measured Intensity' in exp_df.columns:
                        # Interpolate simulation data to match experimental frequencies
                        from scipy.interpolate import interp1d
                        sim_interp = interp1d(freq_data['frequencies'], freq_data['responses'], 
                                           bounds_error=False, fill_value='extrapolate')
                        
                        sim_at_exp_freq = sim_interp(exp_df['Frequency (Hz)'])
                        
                        # Calculate correlation coefficient
                        correlation = np.corrcoef(sim_at_exp_freq, exp_df['Measured Intensity'])[0, 1]
                        
                        # Display metrics
                        st.metric("Correlation Coefficient", f"{correlation:.3f}")
                        
                        # Calculate mean absolute error
                        mae = np.mean(np.abs(sim_at_exp_freq - exp_df['Measured Intensity']))
                        st.metric("Mean Absolute Error", f"{mae:.3f}")
            else:
                st.info("Run a simulation and ensure your data includes frequency measurements to enable comparison.")
                
            # Export options
            st.subheader("Export Results")
            
            # Generate combined data for export
            if st.session_state.simulation.resonance_field is not None:
                if st.button("Export Comparison Data"):
                    # Create a combined DataFrame with both simulation and experimental results
                    # This is a placeholder - you would implement the actual data combination logic
                    st.success("Data export functionality will be implemented in a future release.")
                    
                    # Display what the exported data would look like
                    st.download_button(
                        label="Download CSV (Preview)",
                        data=st.session_state.experimental_data.to_csv(index=False),
                        file_name="sonares_comparison_data.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Upload experimental data to enable comparative analysis.")

# Tab 6: Database Records
with tab6:
    st.header("Database Records")
    
    st.markdown("""
    This tab provides access to the database of saved simulation results and experimental data.
    Build your empirical foundation by storing, comparing, and analyzing multiple simulations.
    """)
    
    # Check if database is available
    if db_available:
        # Create tabs for different record types
        db_tab1, db_tab2, db_tab3 = st.tabs([
            "Simulation Records", 
            "Experimental Data",
            "Save Current Simulation"
        ])
        
        # Tab for viewing simulation records
        with db_tab1:
            st.subheader("Simulation Records")
            
            try:
                # Get simulation records from database
                simulation_records = db_handler.get_simulation_results(limit=50)
                
                if simulation_records and len(simulation_records) > 0:
                    # Convert records to dataframe for display
                    records_data = []
                    for record in simulation_records:
                        records_data.append({
                            'ID': record.id,
                            'Name': record.name,
                            'Material': record.material_name,
                            'Frequency (Hz)': record.frequency,
                            'Sources': record.source_count,
                            'Max Intensity': f"{record.max_intensity:.3f}",
                            'Medium': record.medium,
                            'Date': record.created_at.strftime('%Y-%m-%d %H:%M')
                        })
                    
                    # Create dataframe and display
                    records_df = pd.DataFrame(records_data)
                    st.dataframe(records_df, use_container_width=True)
                    
                    # Add functionality to view details of a specific record
                    selected_record_id = st.selectbox(
                        "Select a record to view details",
                        [record.id for record in simulation_records],
                        format_func=lambda x: f"Record #{x}"
                    )
                    
                    if selected_record_id:
                        # Find the selected record
                        selected_record = next((r for r in simulation_records if r.id == selected_record_id), None)
                        
                        if selected_record:
                            st.subheader(f"Record Details: {selected_record.name}")
                            
                            # Display record details
                            st.json({
                                'Material': selected_record.material_name,
                                'Frequency': selected_record.frequency,
                                'Source Count': selected_record.source_count,
                                'Source Arrangement': selected_record.source_arrangement,
                                'Medium': selected_record.medium,
                                'Reflection Coefficient': selected_record.reflection_coefficient,
                                'Maximum Intensity': selected_record.max_intensity,
                                'Resonance Factor': selected_record.resonance_factor,
                                'Resonance Hotspots': selected_record.resonance_hotspots,
                                'Notes': selected_record.notes if selected_record.notes else "None",
                                'Created': selected_record.created_at.strftime('%Y-%m-%d %H:%M:%S')
                            })
                else:
                    st.info("No simulation records in the database. Save a simulation to get started.")
            except Exception as e:
                st.error(f"Error retrieving database records: {e}")
        
        # Tab for viewing experimental data records
        with db_tab2:
            st.subheader("Experimental Data Records")
            
            try:
                # Get experimental data records from database
                exp_records = db_handler.get_experimental_data(limit=50)
                
                if exp_records and len(exp_records) > 0:
                    # Convert records to dataframe for display
                    exp_data = []
                    for record in exp_records:
                        exp_data.append({
                            'ID': record.id,
                            'Name': record.name,
                            'Material': record.material_name,
                            'Frequency Range': f"{record.frequency_range_min if record.frequency_range_min else 'N/A'} - {record.frequency_range_max if record.frequency_range_max else 'N/A'} Hz",
                            'Data Points': record.data_points,
                            'Date': record.created_at.strftime('%Y-%m-%d %H:%M')
                        })
                    
                    # Create dataframe and display
                    exp_df = pd.DataFrame(exp_data)
                    st.dataframe(exp_df, use_container_width=True)
                    
                    # Add functionality to view details of a specific experimental record
                    selected_exp_id = st.selectbox(
                        "Select experimental data to view",
                        [record.id for record in exp_records],
                        format_func=lambda x: f"Experiment #{x}"
                    )
                    
                    if selected_exp_id:
                        # Get the data as a dataframe
                        exp_df = db_handler.get_experimental_data_as_df(selected_exp_id)
                        
                        if exp_df is not None:
                            st.subheader("Experimental Data")
                            st.dataframe(exp_df, use_container_width=True)
                            
                            # Plot the data if it contains frequency information
                            if 'Frequency (Hz)' in exp_df.columns and 'Measured Intensity' in exp_df.columns:
                                st.subheader("Data Visualization")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(exp_df['Frequency (Hz)'], exp_df['Measured Intensity'], 
                                       marker='o', linestyle='-', color='green')
                                ax.set_xlabel('Frequency (Hz)')
                                ax.set_ylabel('Measured Intensity')
                                ax.set_title('Experimental Data')
                                ax.grid(True, linestyle='--', alpha=0.7)
                                st.pyplot(fig)
                        else:
                            st.warning("Could not load experimental data")
                else:
                    st.info("No experimental data records in the database. Upload and save experimental data to get started.")
            except Exception as e:
                st.error(f"Error retrieving experimental data: {e}")
        
        # Tab for saving current simulation
        with db_tab3:
            st.subheader("Save Current Simulation")
            
            # Form for saving the current simulation
            with st.form(key='save_simulation_form'):
                simulation_name = st.text_input("Simulation Name", 
                                              value=f"Simulation - {st.session_state.selected_material} at {st.session_state.selected_frequency}Hz")
                
                simulation_notes = st.text_area("Notes", 
                                              placeholder="Enter any notes about this simulation...")
                
                save_button = st.form_submit_button("Save to Database")
                
                if save_button:
                    if st.session_state.simulation.resonance_field is not None:
                        # Attempt to save the simulation
                        success = st.session_state.simulation.save_simulation_to_database(
                            name=simulation_name,
                            notes=simulation_notes
                        )
                        
                        if success:
                            st.success("Simulation successfully saved to database!")
                        else:
                            st.error("Error saving simulation to database. Check logs for details.")
                    else:
                        st.warning("Run a simulation before saving.")
            
            # Section for saving experimental data
            st.subheader("Save Experimental Data")
            
            if 'experimental_data' in st.session_state and st.session_state.experimental_data is not None:
                with st.form(key='save_experimental_form'):
                    exp_name = st.text_input("Dataset Name", 
                                           value=f"Experiment - {st.session_state.selected_material}")
                    
                    exp_notes = st.text_area("Notes", 
                                          placeholder="Enter any notes about this experimental data...")
                    
                    exp_save_button = st.form_submit_button("Save to Database")
                    
                    if exp_save_button:
                        # Attempt to save the experimental data
                        success = st.session_state.simulation.save_experimental_data_to_database(
                            name=exp_name,
                            data_df=st.session_state.experimental_data,
                            notes=exp_notes
                        )
                        
                        if success:
                            st.success("Experimental data successfully saved to database!")
                        else:
                            st.error("Error saving experimental data. Check logs for details.")
            else:
                st.info("Upload experimental data before saving to the database.")
    else:
        st.warning("Database functionality is not available. Check your database configuration.")

# Footer
st.markdown("---")
st.markdown("""
**SONARES - Sonic Resonance Evaluation System**

This software simulates acoustic resonance effects on various materials in a virtual 3D environment. 
The simulation uses a sophisticated model of wave physics to demonstrate how different materials respond 
to acoustic stimulation at various frequencies and configurations.

*This is a scientific simulation tool for research and educational purposes.*
""")

# Display status message
if st.session_state.needs_update:
    st.warning("Simulation parameters changed. Click 'Run Simulation' to update visualizations.")
