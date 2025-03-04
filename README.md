<img src="graphics/virpbanner.png" width="870">

# `virp`: VIRtual cell generation by Permutation
 `virp` is a code for the fast generation of a virtual cell from a crystal structure (in CIF format) containing site disorder. It is named after Singapore's first superhero, VR Man, whose superpower is "Virping". The show was a flop, but we are still proud of him. 
 
 This project is inspired by the `Supercell` code of Okhotnikov, Charpentier and Cadars (<i>J. Cheminform. <b>8</b>, 17</i>), which formed the basis of our fast virtual cell generation algorithm, as well as the `aflow++` framework (<i>Comput. Mater. Sci. <b>217</b>, 111889</i>), for the statistical postprocessing of materials properties.

 ## Theory
 First principles materials simulations and computational materials databases are adapted to represent ideal, ordered crystalline materials. However, to treat stoichiometric tuning in chemical compositions, defect-ordered compounds, metal alloys, or many naturally-occuring minerals, site disorder has to be considered. This has been variously treated with special quasirandom structures, cluster expansion, or statistical combination of simulation results from a library of quasirandom structures (here, we call them <i>virtual cells</i>). The configurational space of virtual cells and their simulated properties can be sampled with a relatively small sample set <i>(proof incoming)</i>. The prediction of structural parameters and materials properties is expedited using machine-learned force fields (e.g. CHGNET) and band gap prediction models (`matgl`). 

 ## Installation
 `pip install git+https://github.com/andypaulchen/virp.git`<br>
 Update to latest release: uninstall and re-install<br>
 <b>NEW!</b> Install with `pip install virp`

 ## Building a database
 The root directory has a folder (`session`) which holds the python scripts which build a library of virtual cells (`generate.py`) and postprocessing scripts (`connectivity.py` and `properties.py`). After each script is run, the results are saved as `.csv` files.

 1. To prepare for a session, copy the `session` folder in your workspace and place the `.cif` files you want to process (make virtual cells + postprocessing) in the subfolder `_disordered_cifs`. Feel free to rename `session` folder to something more identifiable

 2. Run `generate.py` to create a supercell and (by default) 400 virtual cells.
    - after this step, a structure subfolder (e.g. `structure`) is created in `session` for each `structure.cif` file in `_disordered_cifs`, with the same name. Inside this folder is a supercell CIF and folders for structure-optimized (`stropt`) and non-structure-optimized virtual cells (`no_stropt`). The details of this run is recorded in `virp_session_summary.csv`.

 3. Run `connectivity.py` for atomic connectivity post-processing
    - after this step, the results are written to `connectivity.csv` and `scatterplot.png` under `stropt` and `no_stropt`.

 4. Run `properties.py` to predict materials properties. This is performed on `stropt` subfolders only.
    - after this step, the results are written to `virtual_properties.csv` in the `structure` subfolder.

 In summary, this is what a session looks like after all three routines have completed:

 <img src="graphics/operation.png" width="870">

 The folder `database_demo` contains `Postprocessing Demo.ipynb`, which concatenates individual sessions into a database (`database.csv`), and conducts postprocessing routines.

 ## Versions and changelog
 `v0.1.1`: first workable code, with function to generate a virtual cell. <br>
 `v0.2.1`: added enumeration function <br>
 `v0.2.2`: enumeration can be imported now (fix) <br>
 `v0.3.0`: you can now make a batch of virtual cells<br>
 `v0.4.3`: added tools to build a database<br>
 `v1.1.0`: major user-friendliness updates, database compilation<br>
 `v1.2.0`: you can now store log files

 ## Debugging and support
 The `virp` code has been tested on a limited number of platforms, so far Windows and Linux. If you are running into any problems during operation, please hound me (Andy Paul Chen) at la.vache.qui.vit(at)gmail.com, and I will try my best to help.
 
