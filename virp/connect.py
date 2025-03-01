from poshcar.disorder import VirtualLibrary, cif2vasp_occ
from sklearn.linear_model import LinearRegression
from pymatgen.core.structure import Structure
from chgnet.model.model import CHGNet
from chgnet.model import StructOptimizer
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Ancillary Functions
#------------------------------------------------------------------------------------------------------------

# Function to format chemical formula with subscripts
def format_formula(formula):
    return ''.join([f'{char}' if not char.isdigit() else f'$_{{{char}}}$' for char in formula])


# User Functions
#------------------------------------------------------------------------------------------------------------

def ConnectivityAnalysis(folder_path = "_disordered_cifs"):
    disordered_cifs = Path(folder_path) # convert to path
    # Load CHGNET (for loops, so we don't do it more than once)
    cn = CHGNet.load()

    for cif_file in disordered_cifs.glob("*.cif"):
        header = cif_file.stem  # Extract filename without extension
        jobdone_path = Path(header) / "_JOBDONE"

        # Proceed only if _JOBDONE exists
        if jobdone_path.exists():
            structure = Structure.from_file(cif_file)
            chemical_formula = structure.composition.reduced_formula
            formatted_formula = format_formula(chemical_formula)
            print("Processing target data... ")
            targetdata = cif2vasp_occ(str(cif_file), verbose=False)

            for branch in ["stropt", "no_stropt"]:
                print(f"\nBranch: {branch}")
                branch_path = Path(header) / branch
                summary_df = VirtualLibrary(branch_path, targetdata, pauling_weight=1, chgnet=cn, structure_opt=False, verbose=False)[1]
                summary_df.to_csv(branch_path / "connectivity.csv")

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
                plt.figure(figsize=(4, 4))
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
                plt.legend()

                # Save the plot as a .png file
                plt.savefig(branch_path / "scatterplot.png", dpi=300, bbox_inches='tight')
                plt.close()