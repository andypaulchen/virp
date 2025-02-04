from virp.matprop import VirtualCellProperties, ExpectationValues
import os

for filename in os.listdir("_disordered_cifs"):
    # extract header string
    if filename.endswith(".cif"):  # Check if the file has a .cif extension
        header = os.path.splitext(os.path.basename(filename))[0]
        if os.path.exists(os.path.join(header, "_JOBDONE")):
            # note: operation on structure-optimized cells only
            VirtualCellProperties(os.path.join(header, "stropt"), os.path.join(header, "virtual_properties.csv"))

# For expectation values post-processing step, run
#ExpectationValues(csv_path, temperature)