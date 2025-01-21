# database.py

# External Imports
from pymatgen.io.cif import CifParser # write pymatgen structure to cif
from pathlib import Path
from tqdm import tqdm
import os

# Internal Imports
#from enumerate import *

def DisorderQuery(folder_path):
    """
    Process all CIF files in a folder to check for partial occupancy.
    Displays a progress bar and summary statistics.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing CIF files
    threshold : float, optional
        Occupancy threshold for checking partial occupancy
    
    Returns:
    --------
    dict
        Dictionary with CIF filenames as keys and their analysis results as values
    """
    folder = Path(folder_path)
    
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder_path}")
    
    results = {}
    cif_files = list(folder.glob("*.cif"))
    
    # Initialize counters
    total_files = len(cif_files)
    files_with_partial = 0
    files_without_partial = 0
    error_files = 0
    
    # Process each CIF file with progress bar
    for cif_file in tqdm(cif_files, desc="Processing CIF files", unit="file"):
        try:
            result = is_site_disordered(str(cif_file))
            results[cif_file.name] = result
            
            # Update counters silently
            if result["has_partial"]:
                files_with_partial += 1
            else:
                files_without_partial += 1
                
        except Exception as e:
            error_files += 1
            results[cif_file.name] = {"error": str(e)}
    
    # Print final summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Total CIF files processed: {total_files}")
    print(f"Files with partial occupancy: {files_with_partial} ({files_with_partial/total_files*100:.1f}%)")
    print(f"Files without partial occupancy: {files_without_partial} ({files_without_partial/total_files*100:.1f}%)")
    if error_files > 0:
        print(f"Files with errors: {error_files} ({error_files/total_files*100:.1f}%)")
    
    return results



def is_SiteDisordered(cif_path):
    """
    Check if a CIF file contains sites with partial occupancy.
    
    Parameters:
    -----------
    cif_path : str
        Path to the CIF file
    threshold : float, optional
        Occupancy threshold below which a site is considered partially occupied
        Default is 1.0 (fully occupied)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - has_partial: bool, whether partial occupancy was found
        - partial_sites: list of tuples (site index, species, occupancy)
    """
    # Verify file exists
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    # Parse the CIF file
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]
    
    # Initialize results
    partial_sites = []
    
    # Check each site in the structure
    for i, site in enumerate(structure.sites):
        species_dict = site.species.as_dict()
        
        # Check occupancy for each species on the site
        for element, occupancy in species_dict.items():
            if occupancy < 1.0:
                partial_sites.append((i, element, occupancy))
    
    result = {
        "has_partial": len(partial_sites) > 0,
        "partial_sites": partial_sites
    }
    
    return result