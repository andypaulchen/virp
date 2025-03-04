from virp.connect import ConnectivityAnalysis
import warnings
import sys

# Redirect stdout to a file
sys.stdout = open("out_connectivity.log", "w")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    ConnectivityAnalysis()