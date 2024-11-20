# main.py

# External Imports
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher # Important to compare two constructed structures
from pymatgen.io.cif import CifWriter # write pymatgen structure to cif
import random
import math
import re

# Internal Imports
from enumerate import *

def CIFSupercell (inputcif, outputcif, supercellsize):
    # inputcif, outputcif: path to cif file
    # supercellsize: vector of 3 integers

    # Load the structure from a CIF file
    structure = Structure.from_file(inputcif)

    # Define the scaling matrix for the supercell
    # For example, [2, 0, 0], [0, 2, 0], [0, 0, 2] creates a 2x2x2 supercell
    scaling_matrix = [[supercellsize[0], 0, 0], 
                      [0, supercellsize[1], 0], 
                      [0, 0, supercellsize[2]]]

    # Create the supercell
    structure.make_supercell(scaling_matrix)

    # Save the supercell to a new CIF file (optional)
    structure.to(fmt="cif", filename=outputcif)
    print("Supercell created and saved as ", outputcif)

    

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
    

def ShuffleOccupiedSites (outfile, edit_block, edit_name):
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
    print("Disordered site name: ", edit_name)
    numberofelements = len(atomoccpairslist)
    print("- Number of elements in this site: ", numberofelements) # The number of elements in this site = N

    # Keep every Nth line in the edit block
    if numberofelements > 1: edit_block = edit_block[::numberofelements]

    # Randomly shuffle the list
    random.shuffle(edit_block)

    # Assign atoms based on proportion in atomoccpairslist
    numberoflines = len(edit_block)
    print("- Number of sites in supercell: ", numberoflines)

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
        
    print("- Atoms and site assignment (float/rounded): ", atomassignmentlist_float)
    print("- No of filled sites: ", assignment_cumulative_int,"/",len(edit_block))
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


def PermutativeFill(input_file, output_file):
    # Updated regex pattern to capture the second string and the last number
    pattern = re.compile(r'\s*\S+\s+(\S+)\s+1\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+([0-9]+\.[0-9]+)')
    
    # Open the input file to read and the output file to write
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Declare edit space (a series of lines where permutative fill takes place)
        edit_active = False # is thisline in an editing block?
        edit_block = []     # array to store lines in an editing block
        edit_name = ""      # stores the site which forms the edit block
        
        for thisline in infile: # scan through the file
            # Check if the line matches the pattern
            match = pattern.match(thisline)
            
            if match: # we have reached the coordinate block of the .cif file
                # Extract the site name and last number from the match
                second_string = match.group(1)  # This will give you 'Ca1'
                last_number = float(match.group(2)) # The last number
    
                # Decision block
                
                if last_number < 1.0: # Check if the last number is less than 1.0: partial occupancy site
                    if not edit_active: # if first line in an edit block
                        edit_active = True # switch on editing mode
                        edit_name = second_string # What site is being edited
                    else:
                        if not edit_name == second_string: # if a different site is being considered
                            ShuffleOccupiedSites(outfile, edit_block, edit_name) # WRITE EDITING BLOCK TO FILE; this also resets it to []
                            # Re-initialize edit parameters
                            edit_block = []     # array to store lines in an editing block
                            edit_active = True
                            edit_name = second_string
                            
                else: # if no longer partial occupancy site
                    if edit_active: 
                        ShuffleOccupiedSites(outfile, edit_block, edit_name) # WRITE EDITING BLOCK TO FILE
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
                outfile.write(thisline)
                if edit_active: # we have reached the end of the coordinate block
                    edit_active = False # switch off edit mode
                    
                    # WRITE SEQUENCE
                    ShuffleOccupiedSites(outfile, edit_block, edit_name)
                    # Re-initialize edit parameters
                    edit_block = []     # array to store lines in an editing block
                    edit_name = ""      # stores the site which forms the edit block
