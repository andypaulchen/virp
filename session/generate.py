from virp import Session, ML_Relaxer
import warnings
import sys

# Redirect stdout to a file
sys.stdout = open("out_generate.log", "w")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Initialize the ML_Relaxer
    mlrelaxer = ML_Relaxer(calc_name="mace_omat", calc_paths=None, optimizer="LBFGSLineSearch", relax_cell=True)

    # Default Settings
    Session(folder_path = "_disordered_cifs", relaxer=mlrelaxer)

    # Custom Settings (for testing)
    #Session(folder_path = "_disordered_cifs", relaxer=mlrelaxer, mindist = 10, sample_size = 2)