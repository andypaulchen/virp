<img src="graphics/virplogo.png" width="200">

# Virp
 `virp` (VIRtual cell generation by Permutation) is a code for the fast generation of a virtual cell from a crystal structure (in CIF format) containing site disorder. It is named after Singapore's first superhero, VR Man, whose superpower is "Virping". The show was a flop, but we are still proud of him.

 ## Theory
 (To be updated!)

 ## Requirements
`pymatgen`, `chgnet`, and `matgl` (`matgl==1.0.0`; `dgl==1.1.2`)<br>
 __Optional__: You can also use git for the fancy installation. Otherwise, downloading the .py file will do.

 ## Installation
 `pip install git+https://github.com/andypaulchen/virp.git`<br>
 Update to latest release: uninstall and re-install

 ## Building a database
 The root directory has a folder (`session`) which holds the python scripts which build a library of virtual cells (`generate.py`) and postprocessing scripts (`connectivity.py` and `properties.py`). After each script is run, the results are saved as `.csv` files. In summary:
 <img src="graphics/operation.png" width="600">

 ## Versions and changelog
 `v0.1.1`: first workable code, with function to generate a virtual cell. <br>
 `v0.2.1`: added enumeration function <br>
 `v0.2.2`: enumeration can be imported now (fix) <br>
 `v0.3.0`: you can now make a batch of virtual cells<br>
 `v0.4.3`: added tools to build a database
 
