# matprop.py

from pymatgen.core.structure import Structure
import os
import csv
import torch
import matgl
from chgnet.model.model import CHGNet

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

    for filename in os.listdir(folder_path):
        if filename.endswith("stropt.cif"): # evaluate structure-optimized cells only
            filepath = os.path.join(folder_path, filename)
            try:
                # Load the structure
                structure = Structure.from_file(filepath)
                
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
                    "File": filename,
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


import pandas as pd
import numpy as np

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