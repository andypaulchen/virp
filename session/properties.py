from virp.matprop import VirtualCellProperties
from pathlib import Path
import warnings
import sys

# Redirect stdout to a file
sys.stdout = open("out_properties.log", "w")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    disordered_cifs = Path("_disordered_cifs")  # Define the folder path

    for cif_file in disordered_cifs.glob("*.cif"):  # Iterate over all .cif files
        header = cif_file.stem  # Extract filename without extension
        jobdone_path = Path(header) / "_JOBDONE"  # Path to _JOBDONE file

        if jobdone_path.exists():  # Check if _JOBDONE exists
            VirtualCellProperties(Path(header) / "stropt", Path(header) / "virtual_properties.csv")