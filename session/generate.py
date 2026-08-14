from virp import Session, ML_Relaxer
import warnings
import sys

# Redirect stdout to a file
sys.stdout = open("out_generate.log", "w")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Initialize the ML_Relaxer
    #mlrelaxer = ML_Relaxer(calc_name="mace_omat", calc_paths=None, optimizer="LBFGSLineSearch", relax_cell=True)

    # Default Settings
    Session()

    # Custom Settings (for testing)
    #Session(folder_path = "", min_length=15.0, max_atoms=1000, length_floor=3.0, sample_size=400, relaxer = None)