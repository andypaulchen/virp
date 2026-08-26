"""
Motif and energy extraction from a folder of CIF structures.

Section 1 (pymatgen): tabulate each site's local coordination "motif"
(center element + neighbor composition [+ geometry]) and aggregate into a
per-structure census — one row per structure, one column per motif type,
values = count or proportion.

Section 2 (MACE): run a MACE-MP interatomic potential over the same CIF
folder to get a per-structure energy, as a target property for downstream
analysis (e.g. shap_motifs() in motif_shap_pipeline.py).

Both folder-level builders (motifs_census_folder, mace_energies_to_csv)
write to disk *and* return a DataFrame, and key on the structure's file
stem ("file_name" / "Name"), so their outputs merge directly without extra
glue code.
"""

import glob
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from ase import Atoms
from ase.io import read
from mace.calculators import mace_mp
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from tqdm.auto import tqdm


# ==============================================================
# User Functions (motifs)
# ==============================================================

def motifs(
    cif_path: str,
    geometry_analysis: bool = False,
    x_diff_weight: float = 0.0,
) -> pd.DataFrame:
    """Build a per-site neighbor-species count table ("cliqueration") for one CIF.

    x_diff_weight is passed straight through to CrystalNN. It's a "prefer
    opposites" dial for picking each atom's neighbors: turned up, CrystalNN
    favors bonds between different elements (e.g. O-H) over bonds between
    the same element (e.g. H-H), even when the same-element atom is
    geometrically closer -- strongly enough that a same-element contact
    can be the single largest raw Voronoi neighbor and still get pushed
    out of the reported coordination shell entirely.

    Default here is 0: neighbors are picked by distance/geometry alone,
    whatever the elements involved -- what this module characterizes is
    the cell as generated (random fills, defects, and all), not an
    idealized chemical picture of it. That's also pymatgen's own
    documented setting for "pure geometric matching, disregarding atomic
    identity."

    pymatgen's own default is 3.0, tuned for identifying "chemically
    sensible" bonding motifs in ordered, relaxed inorganic crystals --
    pass x_diff_weight=3.0 (or elsewhere in that range) if that's what a
    given analysis calls for instead.
    """
    structure = Structure.from_file(cif_path)
    cnn = CrystalNN(x_diff_weight=x_diff_weight)

    # Ordinal labels per species, e.g. Zn1, Zn2, ..., V1, ..., O1, ...
    species_counter = {}
    row_labels = []
    for site in structure:
        el = site.specie.symbol
        species_counter[el] = species_counter.get(el, 0) + 1
        row_labels.append(f"{el}{species_counter[el]}")

    species_set = sorted(species_counter.keys())
    cliqueration = pd.DataFrame(0, index=row_labels, columns=species_set)

    for i, site in enumerate(structure):
        nn_info = cnn.get_nn_info(structure, i)
        for neighbor in nn_info:
            neighbor_el = neighbor["site"].specie.symbol
            cliqueration.loc[row_labels[i], neighbor_el] += 1

    if geometry_analysis:
        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure=structure)
        struct_env = lgf.compute_structure_environments(only_cations=False)
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
        light_se = LightStructureEnvironments.from_structure_environments(
            strategy=strategy, structure_environments=struct_env
        )
        geometries = []
        for i in range(len(structure)):
            ce = light_se.coordination_environments[i]
            geometries.append(ce[0]["ce_symbol"] if ce else "undetermined")
        cliqueration["geometry"] = pd.Series(geometries, index=row_labels, dtype="string")

    return cliqueration


def motifs_census(cliqueration: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-site cliqueration table into named-motif counts/proportions."""
    species_cols = [c for c in cliqueration.columns if c != "geometry"]
    names = []
    for row_label, row in cliqueration.iterrows():
        center_el = re.match(r"[A-Za-z]+", row_label).group()
        # keep only nonzero neighbor counts, sorted by atomic number
        neighbors = [(el, int(row[el])) for el in species_cols if row[el] > 0]
        neighbors.sort(key=lambda x: Element(x[0]).Z)
        neighbor_str = "".join(f"{el}{count if count > 1 else ''}" for el, count in neighbors)
        name = f"{center_el}-{neighbor_str}"
        if "geometry" in cliqueration.columns:
            name = f"{name} ({row['geometry']})"
        names.append(name)

    census = pd.Series(names).value_counts().reset_index()
    census.columns = ["clique_type", "count"]
    center_el = census["clique_type"].str.extract(r"^([A-Za-z]+)-")[0]
    census["proportion"] = census["count"] / census.groupby(center_el)["count"].transform("sum")

    # sort rows by atomic number of the center element
    census["_z"] = center_el.map(lambda el: Element(el).Z)
    census = census.sort_values("_z").drop(columns="_z").reset_index(drop=True)
    return census


def motifs_census_folder(
    folder_path: str,
    use_proportion: bool = True,
    out_csv: Optional[str] = "motifs_census.csv",
    x_diff_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Run motifs() + motifs_census() over every CIF in a folder and assemble a
    wide (structures x motif types) table, keyed by structure name
    ("file_name") to line up with mace_energies_to_csv()'s "Name" column.

    x_diff_weight is forwarded to motifs() -> CrystalNN (see motifs()'
    docstring). Default is 0 (pure distance-based neighbors), so
    same-element contacts -- e.g. H-H clashes -- show up in the census
    instead of being silently outranked by chemically-favored neighbors.
    Pass x_diff_weight=3.0 to restore pymatgen's chemically-weighted
    default for analyses that specifically want idealized bonding motifs.

    Saves to out_csv (pass None to skip writing) and always returns the
    DataFrame, so it can be used standalone or piped straight into
    shap_motifs() via the CSV it just wrote.
    """
    value_col = "proportion" if use_proportion else "count"
    fill_value = 0.0 if use_proportion else 0

    cif_paths = glob.glob(os.path.join(folder_path, "*.cif"))
    rows = {}
    for cif_path in tqdm(cif_paths, desc="Analyzing CIFs"):
        file_name = os.path.splitext(os.path.basename(cif_path))[0]
        cliqueration = motifs(cif_path, x_diff_weight=x_diff_weight)
        census = motifs_census(cliqueration)
        rows[file_name] = census.set_index("clique_type")[value_col]

    result = pd.DataFrame(rows).T.fillna(fill_value)
    result.columns.name = None
    result.index.name = "file_name"
    result = result.reset_index()

    if out_csv:
        result.to_csv(out_csv, index=False)
        print(f"✓ Saved {len(result)} entries to {out_csv}")

    return result


# ==============================================================
# User Functions (total energy calculation)
# ==============================================================
def load_mace_calculator(model: str = "MACE-matpes-r2scan-omat-ft.model", device: str = "cpu"):
    """
    Load a MACE-MP calculator once; reuse the same instance across every CIF
    (loading model weights per-structure would dominate runtime). Kept as an
    explicit call rather than a module-level side effect so importing this
    file doesn't force a model load for callers who only need the motif
    pieces.
    """
    return mace_mp(model=model, device=device)

def mace_energy_from_cif(cif: str, calc = None) -> float:
    """Compute the MACE-predicted potential energy for a single CIF structure."""
    if calc == None:
        calc = mace_mp(model="MACE-matpes-r2scan-omat-ft.model", device="cpu")
    atoms: Atoms = read(cif)  # ASE reads CIF directly
    atoms.calc = calc
    return atoms.get_potential_energy()


def mace_energies_to_csv(
    folder_path: str,
    calc = None,
    out_csv: Optional[str] = "mace_energies.csv",
) -> pd.DataFrame:
    """
    Calculate MACE energies for every CIF in a folder, keyed by structure
    name ("Name") to line up with motifs_census_folder()'s "file_name"
    column. Saves to out_csv (pass None to skip writing) and always returns
    the DataFrame.
    """
    if calc == None:
        calc = mace_mp(model="MACE-matpes-r2scan-omat-ft.model", device="cpu")

    cif_files = sorted(Path(folder_path).glob("*.cif"))
    results = []
    for cif_file in tqdm(cif_files, desc="Calculating MACE", unit="cif"):
        try:
            energy = mace_energy_from_cif(str(cif_file), calc)
        except Exception as e:
            print(f"\nFailed: {cif_file.name} ({e})")
            energy = None
        results.append({"Name": cif_file.stem, "MACE-MATPES": energy})

    result = pd.DataFrame(results)

    if out_csv:
        result.to_csv(out_csv, index=False)
        print(f"✓ Saved {len(result)} entries to {out_csv}")

    return result