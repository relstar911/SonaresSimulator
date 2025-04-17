import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Batch Testing Demo")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Material Comparison", "Parameter Optimization", "Batch Report"])

# Tab 1: Material Comparison
with tab1:
    st.header("Material Comparison")
    
    st.markdown("### Select Materials to Compare")
    
    # Mock data for materials
    all_materials = ["Wood Pine", "Concrete", "Clay Brick", "Steel", "Glass", "Plaster"]
    selected_materials = st.multiselect(
        "Materials",
        options=all_materials,
        default=[all_materials[0]] if all_materials else []
    )
    
    if not selected_materials:
        st.warning("Please select at least one material for comparison")
    else:
        # Run batch test
        if st.button("Run Comparison"):
            with st.spinner("Running batch test..."):
                st.success(f"Completed comparison for {len(selected_materials)} materials")
                
                # Create mock comparison plots for each material
                for material in selected_materials:
                    st.markdown(f"### {material}")
                    
                    # Create metrics display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Correlation", f"{0.85:.3f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{0.12:.3f}")
                    with col3:
                        st.metric("Mean Squared Error", f"{0.02:.3f}")
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Sample data
                    x = np.linspace(50, 1000, 20)
                    y1 = np.sin(x/100) + 0.5
                    y2 = y1 + 0.1 * np.random.randn(len(x))
                    
                    ax.plot(x, y1, 'b-', label="Simulation")
                    ax.scatter(x, y2, color='red', s=50, alpha=0.7, label="Experimental")
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Intensity')
                    ax.set_title(f'Simulation vs. Experimental Data: {material}')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    
                    st.pyplot(fig)
                
                # Create summary table
                st.markdown("### Comparison Summary")
                summary_data = []
                for material in selected_materials:
                    summary_data.append({
                        'Material': material,
                        'Correlation': f"{0.85:.3f}",
                        'MAE': f"{0.12:.3f}",
                        'MSE': f"{0.02:.3f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

# Tab 2: Parameter Optimization
with tab2:
    st.subheader("Parameter Optimization")
    
    st.markdown("""
    Find optimal simulation parameters to match experimental data.
    The system will test multiple parameter combinations to find the best match.
    """)
    
    # Material selection for optimization
    material_to_optimize = st.selectbox(
        "Select material to optimize",
        options=["Wood Pine", "Concrete", "Clay Brick", "Steel", "Glass", "Plaster"]
    )
    
    # Define parameter ranges
    st.markdown("### Parameter Ranges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        reflection_values = st.multiselect(
            "Reflection coefficients to test",
            options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            default=[0.0, 0.3, 0.6, 0.9]
        )
        
    with col2:
        medium_values = st.multiselect(
            "Media to test",
            options=["air", "water", "wood", "concrete", "steel"],
            default=["air"]
        )
    
    # Metric selection
    optimization_metric = st.radio(
        "Optimization metric",
        options=["correlation", "mae", "mse"],
        index=0,
        horizontal=True
    )
    
    # Run optimization
    if st.button("Run Optimization"):
        if not reflection_values or not medium_values:
            st.warning("Please select at least one value for each parameter")
        else:
            with st.spinner("Running parameter optimization..."):
                st.success("Optimization complete!")
                
                # Display optimal parameters
                st.markdown("### Optimal Parameters")
                
                # Mock optimal parameters
                opt_params = {
                    'medium': 'air',
                    'reflection_coefficient': 0.3
                }
                metrics = {
                    'correlation': 0.92,
                    'mae': 0.08,
                    'mse': 0.01
                }
                
                # Create parameter table
                params_df = pd.DataFrame({
                    'Parameter': list(opt_params.keys()),
                    'Optimal Value': list(opt_params.values())
                })
                
                st.dataframe(params_df, use_container_width=True)
                
                # Display metrics
                st.markdown("### Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correlation", f"{metrics['correlation']:.3f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{metrics['mae']:.3f}")
                with col3:
                    st.metric("Mean Squared Error", f"{metrics['mse']:.3f}")
                
                # Apply optimal parameters to current simulation
                if st.button("Apply Optimal Parameters"):
                    st.success("Applied optimal parameters to current simulation!")
                    st.info("Return to other tabs to see the updated simulation results.")

# Tab 3: Batch Report
with tab3:
    st.subheader("Batch Testing Report")
    
    st.markdown("""
    Generate a comprehensive report of material performance across different tests.
    This helps identify the best materials for specific acoustic applications.
    """)
    
    if st.button("Generate Comprehensive Report"):
        with st.spinner("Running batch tests and generating report..."):
            # Generate mock report
            materials = ["Wood Pine", "Concrete", "Clay Brick", "Steel", "Glass", "Plaster"]
            correlations = [0.92, 0.87, 0.79, 0.95, 0.88, 0.83]
            maes = [0.08, 0.13, 0.21, 0.05, 0.12, 0.17]
            mses = [0.01, 0.02, 0.04, 0.01, 0.02, 0.03]
            
            # Create dataframe
            report_data = []
            for i, material in enumerate(materials):
                report_data.append({
                    'Material': material,
                    'Correlation': correlations[i],
                    'MAE': maes[i],
                    'MSE': mses[i]
                })
            
            # Create dataframe
            report_df = pd.DataFrame(report_data)
            
            st.success("Report generated successfully!")
            
            # Display the report
            st.markdown("### Material Performance Report")
            st.dataframe(report_df, use_container_width=True)
            
            # Create visualization
            st.markdown("### Correlation Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(report_df['Material'], report_df['Correlation'], color='skyblue')
            ax.set_xlabel('Material')
            ax.set_ylabel('Correlation with Experimental Data')
            ax.set_title('Material Performance Comparison')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download option
            st.download_button(
                label="Download Report CSV",
                data=report_df.to_csv(index=False),
                file_name="sonares_material_report.csv",
                mime="text/csv"
            )