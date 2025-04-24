# enumerate.py: counts possible permutations and combinations for atom filling in disordered sites

from itertools import product
from math import factorial, prod
import numpy as np
import pandas as pd
import re

# Ancillary Functions
#------------------------------------------------------------------------------------------------------------

def format_integer(num, prec = 6):
    return np.format_float_scientific(num, precision=prec) if num >= 10**prec else str(num)


def discretize_floats(arr):
    # Store possible discretizations for each float
    discretizations = []
    
    for num in arr:
        if num % 1 == 0.5:  # Equidistant case
            lower = int(num // 1)  # Round down
            upper = lower + 1      # Round up
            discretizations.append([lower, upper])
        else:
            discretizations.append([round(num)])  # Standard rounding
    
    # Generate all combinations of discretizations
    all_discretizations = [list(discretization) for discretization in product(*discretizations)]
    
    return all_discretizations
    

def remove_duplicate_sublists(lst):
    seen = set()
    unique_sublists = []
    for sublist in lst:
        sublist_tuple = tuple(sublist)
        if sublist_tuple not in seen:
            seen.add(sublist_tuple)
            unique_sublists.append(sublist)
    return unique_sublists


def enumerate_site(N, compositions, verbose = True):
    # Enumerate combination of enumerations by disordered site
    # N: number of sites
    # compositions: [float], partition fractions adding up to < 1
    if sum(compositions) > 1: print("Error: Compositions add up to more than 100%: ", compositions) # This no make sense (failsafe)
    else:
        if sum(compositions) < 1: compositions.append(1-sum(compositions)) # include vacancies in permutation
        partitions = []
        for i in range(len(compositions)):
            partitions.append(sum(compositions[:i+1]))
        partN = [i*N for i in partitions]
        
        # initialize snapping
        total_combinations = factorial(N)
        if verbose: print("- Raw permutations: ", format_integer(total_combinations), "(", N, "!)")

        # discretize floats
        all_snaps = discretize_floats(partN)
        # assign at least 1 atom per element
        for snap in all_snaps:
            for index in range(len(snap)):
                if index > 0:
                    if snap[index] == snap[index-1]:
                        if verbose and (snap[index] + 1 >= N): print("Error: Choose a bigger supercell!")
                        else: snap[index] += 1
        # remove duplicates in all_snaps
        all_snaps = remove_duplicate_sublists(all_snaps)

        # for each snap, calculate number of combinations
        allcombinations = 0
        for snap in all_snaps:
            if verbose: print("- Snap: ", snap)
            combination = total_combinations
            for index in range(len(snap)):
                if index == 0: n = snap[index]
                else: n = snap[index]-snap[index-1]
                combination /= factorial(n)
            thiscombination = int(combination)
            allcombinations += thiscombination
            if verbose: print("- No. of combinations: ", format_integer(thiscombination)) 

        return all_snaps, allcombinations
    

def get_site_combination(edit_block, edit_name):
    # Auxiliary function which, outside of the enumerate structure routine, will make no sense whatsoever

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
    numberoflines = len(edit_block)
    print("- Number of sites in supercell: ", numberoflines)
    print("- Element and occupancy: ", atomoccpairslist) # The number of elements in this site = N
    proportions = [t[1] for t in atomoccpairslist]
    combinations = enumerate_site(numberoflines, proportions)[1]

    return combinations


# User Functions
#------------------------------------------------------------------------------------------------------------

def Enumerate(input_file):
    # Given a SUPERCELL .cif structure, return total possible virtual cells,
    # disregarding symmetry equivalence
    print("Input supercell .cif file: ", input_file)

    # Updated regex pattern to capture the second string and the last number
    pattern = re.compile(r'\s*\S+\s+(\S+)\s+1\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+([0-9]+\.[0-9]+)')
    product_list = [] # list of permutations to include in

    # Open the input file to read and the output file to write
    with open(input_file, 'r') as infile:
        # Declare edit space (as in permutative fill, but without updating)
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
                            product_list.append(get_site_combination(edit_block, edit_name)) # get combinations for site
                            # Re-initialize edit parameters
                            edit_block = []     # array to store lines in an editing block
                            edit_active = True
                            edit_name = second_string
                            
                else: # if no longer partial occupancy site
                    if edit_active: 
                        product_list.append(get_site_combination(edit_block, edit_name)) # get combinations for site
                        # Re-initialize edit parameters
                        edit_active = False # switch off edit mode
                        edit_block = []     # array to store lines in an editing block
                        edit_name = ""      # stores the site which forms the edit block

                if edit_active:
                    # Write the line to the edit block
                    edit_block.append(thisline)
    
    totalcombinations = prod(product_list)
    print("Total number of combinations for", input_file, ": ", format_integer(totalcombinations))
    return totalcombinations


def EquivalentStructures(csv_path):
    # Read CSV and sort by FormationEnergy
    df = pd.read_csv(csv_path)
    number_rows = df.shape[0]
    df_sorted = df.sort_values(by='Total Energy (eV)').reset_index(drop=True)
    
    # Convert FormationEnergy values to strings
    energy_strings = df_sorted['Total Energy (eV)'].astype(str)
    
    # Count exact string matches between adjacent terms
    overlaps = sum(energy_strings.iloc[i] == energy_strings.iloc[i+1] for i in range(len(energy_strings)-1))
    percentage = 100*overlaps/number_rows
    
    print(f"Number of redundant structures: {overlaps}/{number_rows} ({percentage:.2f}%)")
    return overlaps