# main.py

# External Imports
from pymatgen.core.structure import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation

from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
import warnings
import random
import math
import re

# ==============================================================
# Ancillary Functions
# ==============================================================

def round_with_tie_breaker(n):
    # Separate the fractional and integer parts
    fractional_part, integer_part = math.modf(n)
    
    # Check if the fractional part is 0.5
    if abs(fractional_part) == 0.5:
        # Randomly choose to round down or up
        return int(integer_part) + random.choice([0, 1])
    else:
        # Regular rounding for non 0.5 cases
        return round(n)


def ShuffleOccupiedSites (outfile, edit_block, edit_name, verbose = True):
    # Auxiliary function which, outside of the permutative fill routine, will make no sense whatsoever

    # 1. What are the unique elements and occupancies?
    atomoccpairslist = []
    for evalline in edit_block:
        # Split each line into components (using split will automatically handle whitespaces)
        parts = evalline.split()
        atomoccpair = (parts[0], float(parts[-1]))
        if atomoccpair not in atomoccpairslist:
            atomoccpairslist.append(atomoccpair)

    # Display specifications
    if verbose: print("Disordered site name: ", edit_name)
    numberofelements = len(atomoccpairslist)
    if verbose: print("- Number of elements in this site: ", numberofelements) # The number of elements in this site = N

    # Keep every Nth line in the edit block
    if numberofelements > 1: edit_block = edit_block[::numberofelements]

    # Randomly shuffle the list
    random.shuffle(edit_block)

    # Assign atoms based on proportion in atomoccpairslist
    numberoflines = len(edit_block)
    if verbose: print("- Number of sites in supercell: ", numberoflines)

    atomassignmentlist_float = []
    assignment_cumulative = 0
    assignment_cumulative_int = 0
    for atomoccpair in atomoccpairslist:
        # evaluate how many atoms to assign to element in question
        atomassignment_float = atomoccpair[1]*numberoflines
        assignment_cumulative += atomassignment_float
        assignment_int = max(round_with_tie_breaker(assignment_cumulative)-assignment_cumulative_int,1) # assign at least 1 atom
        assignment_cumulative_int += assignment_int
        # tuples for display
        atomassignmentlist_float.append((atomoccpair[0], atomassignment_float, assignment_int))
        
    if verbose: print("- Atoms and site assignment (float/rounded): ", atomassignmentlist_float)
    if verbose: print("- No of filled sites: ", assignment_cumulative_int,"/",len(edit_block))
    edit_block = edit_block[:assignment_cumulative_int]

    # Implement the atom-site assignment in-text
    pointer = 0 # line-by line pointer for edit_block rows
    for this_element in atomassignmentlist_float:
        element_name = this_element[0]
        no_atoms = this_element[2]
        for i in range(no_atoms):
            edit_block[pointer] = re.sub(r'^(\s*)([^\s]+)', r'\1' + element_name, edit_block[pointer])
            pointer += 1

    # Change every occupancy to 1.0
    edit_block = [re.sub(r'([0-9]+\.[0-9]+)\s*$', '1.0', line) + '\n' for line in edit_block]
    
    for writeline in edit_block: outfile.write(writeline)


def ImportICSD(csv_path, qrange = None):
    # Update: this function now moves .cif files from CSV into "_disordered_cifs" 
    # instead of building the database
    
    # Make _disordered_cifs path if not already existing
    Path("_disordered_cifs").mkdir(exist_ok=True)

    # Import CSV database
    db_dis = pd.read_csv(csv_path)
    if qrange == None: qrange = len(db_dis) # if unspecified, process all entries

    # Loop through all .cif files in the folder
    for entry in range(qrange):
        print("Processing entry #",entry)
        try:
            # write structure to temporary cif path
            with open("temp.cif", "w", encoding="utf-8", errors="replace") as f:
                f.write(db_dis['cif'].iloc[entry])

            #Extract metadata: chemical formula
            structure = Structure.from_file("temp.cif")
            formula = structure.composition.reduced_formula
            elements = [str(el.symbol) for el in structure.composition.elements]

            filename = str(db_dis['CollectionCode'].iloc[entry])+"_"+formula+".cif"
            Path("temp.cif").rename(Path("_disordered_cifs") / filename)
            print(f"Writing to .cif file: {filename}")

        except Exception as e:
            print(f"Error processing Entry #{entry}: {e}")


# ==============================================================
# User Functions
# ==============================================================

def Supercell(inputcif, outputcif, min_length=15.0, max_atoms=1000, length_floor=3.0, verbose=True):
    # inputcif, outputcif: path to cif file
    # min_length: minimum length (Angstrom) of the supercell's inscribed cube/box
    # max_atoms: cap on resulting supercell size
    # length_floor: don't shrink min_length below this (Angstrom), avoids infinite/degenerate search
    
    # --- Input: read structure from CIF ---
    structure = Structure.from_file(inputcif)

    if verbose:
        print("Original lattice:")
        print(structure.lattice)
        print("Original angles (alpha, beta, gamma):", structure.lattice.angles)
        print("Number of atoms:", len(structure))
    
    # --- Apply transformation, rebuilding cst each iteration ---
    ortho_structure = None
    no_atoms = max_atoms + 1
    while (min_length >= length_floor) and (no_atoms > max_atoms):
        cst = CubicSupercellTransformation(min_length=min_length, force_90_degrees=False)
        # --- Apply transformation ---
        if verbose: print(f"\nTransforming with minimum length {min_length} Angstroms...")
        ortho_structure = cst.apply_transformation(structure)
        no_atoms = len(ortho_structure)
        if no_atoms > max_atoms:
            print(f"Supercell has {no_atoms} atoms... reducing minimum length to {min_length - 1.0}")
            min_length -= 1.0
            if min_length < length_floor: 
                print("\nError: minimum length below floor")
                return

    # Calclate actual min distance of lattice
    lattice = ortho_structure.lattice
    min_dist_actual = min(
        np.linalg.norm(lattice.get_cartesian_coords(v))
        for v in product([-1, 0, 1], repeat=3)
        if v != (0, 0, 0)
    )
    
    if verbose:
        print("\nOrthogonalized lattice:")
        print(ortho_structure.lattice)
        print("Orthogonalized angles:", ortho_structure.lattice.angles)
        print("Minimum distance (prescribed/actual):", min_length, " / ", min_dist_actual)
        print("Number of atoms:", len(ortho_structure))
        # You can also inspect the transformation matrix that was used
        print("\nTransformation matrix:")
        print(cst.transformation_matrix)
    
    # --- Output: write orthogonalized structure ---
    ortho_structure.to(filename=outputcif)
    if verbose: print("\nSupercell created and saved as", outputcif)

    return min_length, min_dist_actual, cst.transformation_matrix


def PermutativeFill(input_file, output_file, verbose = True):
    # Updated regex pattern to capture the second string and the last number
    pattern = re.compile(r'\s*\S+\s+(\S+)\s+1\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+([0-9]+\.[0-9]+)')

    # Append a line to the input file to avoid EOF issues
    with open(input_file, 'a') as infile:
        infile.write("\n#EOF")  # Append at the end
    
    # Open the input file to read and the output file to write
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Declare edit space (a series of lines where permutative fill takes place)
        edit_active = False # is thisline in an editing block?
        edit_block = []     # array to store lines in an editing block
        edit_name = ""      # stores the site which forms the edit block
        #lines = infile.readlines()  # Read all lines
        
        for thisline in infile: # scan through the file
            # Check if the line matches the pattern
            match = pattern.match(thisline)
            
            if match: # we have reached the coordinate block of the .cif file
                # Extract the site name and last number from the match
                second_string = re.sub(r'_\d+$', '', match.group(1))  # This will give you 'Ca1'
                last_number = float(match.group(2)) # The last number
    
                # Decision block
                
                if last_number < 1.0: # Check if the last number is less than 1.0: partial occupancy site
                    if not edit_active: # if first line in an edit block
                        edit_active = True # switch on editing mode
                        edit_name = second_string # What site is being edited
                    else:
                        if not edit_name == second_string: # if a different site is being considered
                            ShuffleOccupiedSites(outfile, edit_block, edit_name, verbose = verbose) # WRITE EDITING BLOCK TO FILE; this also resets it to []
                            # Re-initialize edit parameters
                            edit_block = []     # array to store lines in an editing block
                            edit_active = True
                            edit_name = second_string
                            
                else: # if no longer partial occupancy site
                    if edit_active: 
                        ShuffleOccupiedSites(outfile, edit_block, edit_name, verbose = verbose) # WRITE EDITING BLOCK TO FILE
                        # Re-initialize edit parameters
                        edit_active = False # switch off edit mode
                        edit_block = []     # array to store lines in an editing block
                        edit_name = ""      # stores the site which forms the edit block

                # Execution block
    
                if edit_active:
                    # Write the line to the edit block
                    edit_block.append(thisline)
    
                else: # edit mode is not active
                    # Write the thisline to the output file
                    outfile.write(thisline)
    
            else: # other lines we are not bothered with
                if thisline.strip() != "#EOF": outfile.write(thisline)
                if edit_active: # we have reached the end of the coordinate block
                    edit_active = False # switch off edit mode
                    
                    # WRITE SEQUENCE
                    ShuffleOccupiedSites(outfile, edit_block, edit_name, verbose = verbose)
                    # Re-initialize edit parameters
                    edit_block = []     # array to store lines in an editing block
                    edit_name = ""      # stores the site which forms the edit block
    
    # Remove the last line (EOF) from the input file
    with open(input_file, 'r') as infile: lines = infile.readlines()  # Read all lines
    if lines and lines[-1].strip() == "#EOF": lines.pop()
    with open(input_file, 'w') as outfile: outfile.writelines(lines)  # Write back without "#EOF"


def MakeVirtualCells(input_cif, min_length=15.0, max_atoms=1000, length_floor=3.0, sample_size=400, relaxer = None):
    """
    Given a disordered .cif file, create an output folder
    containing a number (sample_size) of virtual cells
    
    Args:
        input_cif (str): Path to .cif (disordered)
        min_length (float): Minimum tolerated distance between lattice points in one direction
        max_atoms (int): Maximum number of atoms allowed in the supercell
        length_floor (float): Minimum length for each lattice vector
        sample_size (int): Number of virtual cells to generate (default is 400)
        relaxer (Relaxer): The relaxer to use for structure optimization (default is None)

    Returns:
        void
    """

    # Suppress warnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Make output folder directory
        fname = Path(input_cif).stem
        Path(fname).mkdir(exist_ok=True)  # `exist_ok=True` avoids errors if the directory exists.
        print(f"Directory created at: {fname}")

        header = Path(fname) / fname
        sc_file = str(header) + "_supercell.cif"

        # Make the supercell
        min_dist_prescribed, min_dist_actual, sc_matrix = Supercell(input_cif, sc_file, min_length=min_length, max_atoms=max_atoms, length_floor=length_floor)

        # Create target folders if they don't exist
        stropt_path = Path(fname) / "stropt"
        no_stropt_path = Path(fname) / "no_stropt"
        Path(stropt_path).mkdir(exist_ok=True)  # structure-optimized cells
        Path(no_stropt_path).mkdir(exist_ok=True)  # non-structure-optimized cells

        # Execution
        for i in range(sample_size):
            # Permutative fill only, no structure optimization
            pfill_file_name = fname+"_virtual_"+str(i)+".cif"
            pfill_file = Path(no_stropt_path) / pfill_file_name
            print(str(sc_file), str(pfill_file))
            PermutativeFill(sc_file, pfill_file, verbose = True if i == 0 else False)
            print(f"\rGenerating virtual cell #{i} ({i+1}/{sample_size})", end="", flush=True)

            # Relax
            # 30 Oct 2025: If relaxer is None, don't relax! (was CHGNET)
            if relaxer != None:
                structure = Structure.from_file(pfill_file)
                result = relaxer.relax(structure, verbose=False)
                stropt_file_name = fname+"_virtual_"+str(i)+"_stropt.cif"
                stropt_file = Path(stropt_path) / stropt_file_name 
                result['final_structure'].to(stropt_file)
            else:
                print("\nNo relaxer specified; skipping structure optimization.")
                nostropt_file = Path(stropt_path) / "no_relax"
                nostropt_file.touch()
        
        with open(Path(fname) / "_JOBDONE", 'w') as file: pass # make an empty file signalling completion
        print("\nAll cells generated (see _JOBDONE file).")

        return min_dist_prescribed, min_dist_actual, sc_matrix


def Session(folder_path = "", min_length=15.0, max_atoms=1000, length_floor=3.0, sample_size=400, relaxer = None):
    """
    Given a set of disordered .cif files, create a session
    which generates (optional: relaxes) virtual cells for each .cif file
    
    Args:
        folder_path (str): where the .cif files are located
        mindist (float): minimum tolerated distance between
            lattice points in one direction
        supercell [int,int,int]: multiplicity of supercell
        sample_size (int): Number of virtual cells to generate (default is 400)
        relaxer (StructOptimizer): relaxer object to use for structure optimization 
            (default is None, i.e., no relax)
        
    Returns:
        void
    """
    # If subfolder "_disordered_cifs" does not exist, create it
    if Path("_disordered_cifs").exists() == False:
        Path("_disordered_cifs").mkdir(exist_ok=True)
        # Copy all .cif files from folder_path to _disordered_cifs
        for cif_file in Path(folder_path).glob("*.cif"):
            target_file = Path("_disordered_cifs") / cif_file.name
            if not target_file.exists():
                with open(cif_file, 'r', encoding="utf-8", errors="replace") as src, open(target_file, 'w', encoding="utf-8", errors="replace") as dst:
                    dst.write(src.read())
                print(f"Copied {cif_file} to {target_file}")
    
    # init DataFrame to store results
    data = []
    session_name = Path.cwd().name
    # for run-id
    session_stem = ".".join(session_name.rsplit(".", 1)[:-1])
    ordinal = 1 # for run-id

    # Default relaxer is None (no relax)
    if relaxer == None: 
        relaxer_name = "none"
        print("No relax performed.")
    else:
        relaxer_name = relaxer.calc_name
        print("Using relaxer: ", relaxer_name)

    # Loop through all .cif files in the folder
    for filename in Path("_disordered_cifs").glob("*.cif"):
        print(f"Processing .cif file: {filename}")

        try:
            # Generate virtual cell samples
            min_dist_prescribed, min_dist_actual, sc_matrix = MakeVirtualCells(filename, min_length=min_length, max_atoms=max_atoms, length_floor=length_floor, sample_size=sample_size, relaxer = None)

            # Extract metadata: chemical formula
            structure = Structure.from_file(filename)
            formula = structure.composition.reduced_formula
            elements = [str(el.symbol) for el in structure.composition.elements]

            # Append results to the data list
            data.append({
                "session": session_name,
                "run_id": f"{session_stem}.{ordinal}",
                "filename": Path(filename).stem,
                "formula": formula,
                "elements": elements,
                "supercell size": sc_matrix,
                "image distance (target)": float(min_dist_prescribed),
                "image distance (actual)": float(min_dist_actual),
                "sample size": sample_size,
                "relaxer": relaxer_name,
                "connectivity_done": False,
                "properties_done": False,
                "provenance": None
            })
            ordinal +=1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    output_file = "virp_session_summary.csv"
    df.to_csv(output_file, index = False)
    print(f"Results saved to {output_file}")