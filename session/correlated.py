# Example: Solve a correlated disorder structure using a SAT solver

import warnings
warnings.filterwarnings('ignore')

SOURCE_FOLDER_PATH = #folder containing source cells (random-filled)
SUPERCELL_CIF_PATH = #path to supercell cif file (e.g. In2Te3_full.cif)
MAX_SOLUTIONS = #400 by default, but can be set to a lower number for testing

# MACE energies from source cells
from virp.motifs import mace_energies_to_csv
df_energies = mace_energies_to_csv(folder_path=SOURCE_FOLDER_PATH)

# Motifs from source cells
from virp.motifs import motifs_census_folder
df_motifs = motifs_census_folder(folder_path=SOURCE_FOLDER_PATH)

# Train a SHAP model to rank motifs by their contribution to the MACE energy
from virp.correlated import shap_motifs
ranked_motifs = shap_motifs(df_motifs, df_energies)

# Use the SAT solver to find correlated arrangements of motifs in the supercell
from virp.correlated import sat_solver
sat_solver(SUPERCELL_CIF_PATH, ranked_motifs, max_solutions=MAX_SOLUTIONS, out_dir="_sat_solutions")