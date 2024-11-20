# enumerate: facilitates atom assignment to disordered sites

from itertools import product
from math import factorial

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


def enumerate_permutations(N, compositions):
    # N: number of slots, int
    # compositions: [float], partition fractions adding up to < 1
    if sum(compositions) > 1: print("Error: Compositions add up to more than 100%: ", compositions) # This no make sense
    else:
        if sum(compositions) < 1: compositions.append(1-sum(compositions)) # include vacancies in permutation
        partitions = []
        for i in range(len(compositions)):
            partitions.append(sum(compositions[:i+1]))
        partN = [i*N for i in partitions]
        #display(partN)
        
        # initialize snapping
        total_combinations = factorial(N)
        print("Raw permutations: ", total_combinations, "(", N, "!)")

        # discretize floats
        all_snaps = discretize_floats(partN)
        # assign at least 1 atom per element
        for snap in all_snaps:
            for index in range(len(snap)):
                if index > 0:
                    if snap[index] == snap[index-1]:
                        if snap[index] + 1 >= N: print("Error: Choose a bigger supercell!")
                        else: snap[index] += 1
        # remove duplicates in all_snaps
        all_snaps = remove_duplicate_sublists(all_snaps)

        # for each snap, calculate number of combinations
        allcombinations = 0
        for snap in all_snaps:
            print("Snap: ", snap)
            combination = total_combinations
            for index in range(len(snap)):
                if index == 0: n = snap[index]
                else: n = snap[index]-snap[index-1]
                combination /= factorial(n)
            allcombinations += int(combination)
            print("No. of combinations: ", int(combination)) 

        return all_snaps, allcombinations