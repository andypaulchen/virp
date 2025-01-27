from virp.matprop import VirtualCellProperties, ExpectationValues
import os

for filename in os.listdir("_disordered_cifs"):
    # extract header string
    header = os.path.splitext(os.path.basename(filename))[0]
    # note: operation on structure-optimized cells only
    VirtualCellProperties(os.path.join(header, "stropt"), os.path.join(header, "virtual_properties.csv"))

# For expectation values post-processing step, run
#ExpectationValues(csv_path, temperature)