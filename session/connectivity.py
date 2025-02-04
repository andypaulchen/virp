from poshcar.disorder import VirtualLibrary, cif2vasp_occ
from sklearn.linear_model import LinearRegression
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
import numpy as np
import os

# Loop over files in session
for filename in os.listdir("_disordered_cifs"):
    # Load the .cif file to extract the chemical formula
    if filename.endswith(".cif"):  # Check if the file has a .cif extension
        header = os.path.splitext(os.path.basename(filename))[0]

        # For everything else, check if _JOBDONE exists (virtual cells succeeded in generating)
        if os.path.exists(os.path.join(header, "_JOBDONE")):
            path_to_file = os.path.join("_disordered_cifs", filename)
            structure = Structure.from_file(path_to_file)  
            chemical_formula = structure.composition.reduced_formula
            # Format the chemical formula to include subscripts for numbers
            def format_formula(formula):
                return ''.join([f'{char}' if not char.isdigit() else f'$_{{{char}}}$' for char in formula])
            formatted_formula = format_formula(chemical_formula)

            targetdata = cif2vasp_occ(path_to_file, verbose = False)
            for branch in ["stropt", "no_stropt"]:
                datalist, summary_df = VirtualLibrary(os.path.join(header, branch), targetdata, pauling_weight = 1, structure_opt = False, verbose = False)
                summary_df.to_csv(os.path.join(header, branch, "connectivity.csv"))

                # Plotting segment
                x = np.array(summary_df['BondDiff']).reshape(-1, 1)
                y = np.array(summary_df['FormationEnergy'])

                # Perform linear regression
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)

                # Calculate R² value
                r_sq = model.score(x, y)

                # Sort data for smooth plotting
                sorted_indices = np.argsort(summary_df['BondDiff'])
                x_sorted = np.array(summary_df['BondDiff'])[sorted_indices]
                y_pred_sorted = y_pred[sorted_indices]

                # Scatterplot with best-fit line
                plt.figure(figsize=(4, 4))  # Adjust figure size if needed
                plt.scatter(summary_df['BondDiff'], summary_df['FormationEnergy'], color='blue', alpha=0.7, label='Data')
                plt.plot(x_sorted, y_pred_sorted, color='grey', linestyle='--', linewidth=2, label=f'Fit (R²={r_sq:.2f})', zorder=2)

                # Labels and title
                plt.xlabel('Bonding deviation ΔB$_{{{P5}}}$', fontsize=14)
                plt.ylabel('Formation Energy (eV/at.)', fontsize=14)
                plt.title(f'{formatted_formula}', fontsize=16)

                # Display R² value on the plot
                plt.text(0.05, 0.9, f"R² = {r_sq:.2f}", fontsize=14, color='black', transform=plt.gca().transAxes)

                # Grid and legend
                plt.grid(True, linestyle='--', alpha=0.5)

                # Save the plot as a .png file
                plt.savefig(os.path.join(header, branch, "scatterplot.png"), dpi=300, bbox_inches='tight')