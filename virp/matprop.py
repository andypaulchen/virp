# matprop.py: Materials properties

from pymatgen.core.structure import Structure
from chgnet.model.model import CHGNet
import csv
import torch
import matgl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# User Functions
#------------------------------------------------------------------------------------------------------------

def VirtualCellProperties(folder_path, output_csv):
    # To add: customise the set of properties to evaluate
    """
    Given a folder filled with virtual cells, 
    predict material properties for each virtual cell,
    and write results in a .csv form
    
    Args:
        folder_path (str): Path to folder
        output_csv (str): Path to .csv output
        
    Returns:
        void
    """
    # Load the MEGNet band gap model
    bandgap_model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
    
    # Load the CHGNet model for total energy prediction
    chgnet = CHGNet.load()

    # Initialize data storage
    data = []

    for filename in Path(folder_path).glob("*stropt.cif"):
        # evaluate structure-optimized cells only
        print(Path(filename).stem)
        try:
            # Load the structure
            structure = Structure.from_file(filename)
            
            # Predict total energy
            total_energy = chgnet.predict_structure(structure)['e']

            # Calculate density and convert to float
            density = float(structure.density)

            # Predict band gaps for different methods
            bandgaps = {}
            for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
                graph_attrs = torch.tensor([i])
                bandgap = bandgap_model.predict_structure(structure=structure, state_attr=graph_attrs)
                bandgaps[method] = float(bandgap)
            
            # Append results to data
            data.append({
                "File": Path(filename).stem,
                "Total Energy (eV)": total_energy,
                "Density": density,
                "PBE Bandgap (eV)": bandgaps["PBE"],
                "GLLB-SC Bandgap (eV)": bandgaps["GLLB-SC"],
                "HSE Bandgap (eV)": bandgaps["HSE"],
                "SCAN Bandgap (eV)": bandgaps["SCAN"],
            })
            
            print(f"Processed: {filename}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Write results to CSV
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ["File", "Total Energy (eV)", "Density", "PBE Bandgap (eV)", "GLLB-SC Bandgap (eV)", "HSE Bandgap (eV)", "SCAN Bandgap (eV)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)

    print(f"Results saved to {output_csv}")


def ExpectationValues(csv_path, temperature):
    """
    Calculate Boltzmann-weighted expectation values for all numeric properties
    
    Args:
        csv_path (str): Path to CSV file
        temperature (float): Temperature in Kelvin
        
    Returns:
        tuple: (DataFrame, dictionary of expectation values)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Boltzmann constant in eV/K = 0.00008617
    k_B = 0.0000861733326
    
    # Calculate weights using the Boltzmann distribution formula
    df['weights'] = np.exp(-df['Total Energy (eV)']/(k_B * temperature))
    
    # Calculate total weights
    total_weights = df['weights'].sum()
    
    # Dictionary to store expectation values
    expectation_values = {}
    
    # Get all numeric columns except 'Total Energy (eV)' and 'weights'
    excluded_cols = ['File', 'Total Energy (eV)', 'weights']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    properties = [col for col in numeric_cols if col not in excluded_cols]
    
    # Calculate weighted properties and their expectation values
    for prop in properties:
        weighted_col_name = f'weighted_{prop}'
        df[weighted_col_name] = (df[prop] * df['weights']) / total_weights
        expectation_values[prop] = df[weighted_col_name].sum()
    
    return df, expectation_values


def Histograms(folder_path, output_path=None):
    """
    Read CSV file and display histograms for all columns except 'File' and 'Total Energy (eV)'.
    
    Parameters:
    folder_path (str): Path to the folder containing the CSV file
    """
    # Construct file path
    csv_path = Path(folder_path) / "virtual_properties.csv"
    
    # Check if file exists
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Remove specified columns
        exclude_columns = ['File']
        plot_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Calculate number of rows and columns for subplot grid
        n_plots = len(plot_columns)
        n_cols = 2  # You can adjust this to change the layout
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('Histograms of Virtual Properties', fontsize=16, y=1.02)
        
        # Flatten axes array for easier iteration
        axes_flat = axes.flatten() if n_plots > 1 else [axes]
        
        # Generate histograms for each column
        for idx, (column, ax) in enumerate(zip(plot_columns, axes_flat)):
            ax.hist(df[column].dropna(), bins=30, edgecolor='black')
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
        # Hide any unused subplots
        for idx in range(len(plot_columns), len(axes_flat)):
            axes_flat[idx].set_visible(False)
            
        # Adjust layout
        plt.tight_layout()

        # Save figure if output path is specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Histogram saved to {output_path}")
            
        plt.show()
        
    except Exception as e:
        raise Exception(f"Error processing CSV file: {str(e)}")