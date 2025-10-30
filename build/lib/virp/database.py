# database.py

# External Imports
from pymatgen.io.cif import CifParser # write pymatgen structure to cif
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Query the local database
#------------------------------------------------------------------------------------------------------------

def CompileDatabase(output_path = "database.csv"):
    # Path to current directory
    current_dir = Path.cwd()
    
    # Initialize an empty list to store dataframes
    all_dataframes = []
    
    # Initialize a counter to track how many files we've processed
    files_processed = 0
    
    # Get all subdirectories that start with "S."
    subdirs = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("S.")]
    print(f"Found {len(subdirs)} Virp session subdirectories")
    
    # Loop through each subdirectory
    for subdir in subdirs:
        # Path to the CSV file in this subdirectory
        csv_path = subdir / "virp_session_summary.csv"
        
        # Check if the file exists
        if csv_path.exists():
            try:
                # Read the CSV file into a dataframe
                df = pd.read_csv(csv_path)
                
                # Append this dataframe to our list
                all_dataframes.append(df)
                
                # Increment counter
                files_processed += 1
                print(f"Processed: {csv_path}")
                
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
        else:
            print(f"File not found: {csv_path}")
    
    # If we found any files, concatenate them
    if all_dataframes:
        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Write the combined dataframe to a new CSV file
        output_path = current_dir / Path(output_path)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully combined {files_processed} files into {output_path}")
        print(f"The combined database has {len(combined_df)} rows and {len(combined_df.columns)} columns")
    else:
        print("No files were found to concatenate.")
    
    return combined_df


def ImportDatabase(input_path = "database.csv"):
    # Same as CompileDatabase, but reads the combined CSV file only
    try:
        df = pd.read_csv(input_path)
        return df
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def GetPath(df, run_id):
    # Check if run_id exists
    if run_id in df["run_id"].values:
        subfolder = df.loc[df['run_id'] == run_id, 'session'].iloc[0]
        subsubfolder = df.loc[df['run_id'] == run_id, 'filename'].iloc[0]
        return Path(subfolder) / Path(subsubfolder)
    else:
        print(f"Run ID {run_id} not found in the database.")
        return None


# Query an external database
#------------------------------------------------------------------------------------------------------------

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
            result = is_SiteDisordered(str(cif_file))
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
    if not Path(cif_path).exists(): raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    # Parse the CIF file
    parser = CifParser(cif_path)
    structure = parser.parse_structures(primitive=True)[0]
    
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