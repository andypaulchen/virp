# visualise.py: Visualise a folder of unit cells together in a GIF

from ase.io import read
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

# VESTA-like element color dictionary
VESTA_COLORS = {
    # Default atom colors used by VESTA (jp-minerals.org/vesta), as RGB floats in [0, 1].
    # Source: VESTA's own element color table (elements.ini), cross-checked against
    # pymatgen's bundled VESTA color scheme (pymatgen/vis/ElementColorSchemes.yaml).
    # Comments are color names computed from each RGB triple's hue/lightness, not guesses.
    'H': (1.0, 0.8, 0.8),  # Very Light Pink
    'He': (0.988, 0.91, 0.808),  # Very Light Orange
    'Li': (0.525, 0.875, 0.451),  # Green
    'Be': (0.369, 0.843, 0.482),  # Green
    'B': (0.122, 0.635, 0.059),  # Dark Green
    'C': (0.298, 0.298, 0.298),  # Dark Gray
    'N': (0.69, 0.725, 0.902),  # Light Blue
    'O': (0.996, 0.012, 0.0),  # Red
    'F': (0.69, 0.725, 0.902),  # Light Blue
    'Ne': (0.996, 0.216, 0.71),  # Pink
    'Na': (0.976, 0.863, 0.235),  # Yellow
    'Mg': (0.984, 0.482, 0.082),  # Orange
    'Al': (0.506, 0.698, 0.839),  # Blue
    'Si': (0.106, 0.231, 0.98),  # Blue
    'P': (0.753, 0.612, 0.761),  # Light Pink
    'S': (1.0, 0.98, 0.0),  # Yellow
    'Cl': (0.192, 0.988, 0.008),  # Green
    'Ar': (0.812, 0.996, 0.769),  # Very Light Green
    'K': (0.631, 0.129, 0.965),  # Violet
    'Ca': (0.353, 0.588, 0.741),  # Blue
    'Sc': (0.71, 0.388, 0.671),  # Pink
    'Ti': (0.471, 0.792, 1.0),  # Light Blue
    'V': (0.898, 0.098, 0.0),  # Red
    'Cr': (0.0, 0.0, 0.62),  # Dark Blue
    'Mn': (0.655, 0.031, 0.616),  # Dark Pink
    'Fe': (0.71, 0.443, 0.0),  # Orange
    'Co': (0.0, 0.0, 0.686),  # Dark Blue
    'Ni': (0.718, 0.733, 0.741),  # Gray
    'Cu': (0.133, 0.278, 0.863),  # Blue
    'Zn': (0.561, 0.561, 0.506),  # Gray
    'Ga': (0.62, 0.89, 0.451),  # Green
    'Ge': (0.494, 0.431, 0.651),  # Muted Violet
    'As': (0.455, 0.816, 0.341),  # Green
    'Se': (0.604, 0.937, 0.059),  # Green
    'Br': (0.494, 0.192, 0.008),  # Dark Orange
    'Kr': (0.98, 0.757, 0.953),  # Very Light Pink
    'Rb': (0.439, 0.18, 0.69),  # Violet
    'Sr': (0.0, 1.0, 0.0),  # Green
    'Y': (0.58, 1.0, 1.0),  # Light Cyan
    'Zr': (0.0, 1.0, 0.0),  # Green
    'Nb': (0.451, 0.761, 0.788),  # Cyan
    'Mo': (0.329, 0.71, 0.71),  # Cyan
    'Tc': (0.231, 0.62, 0.62),  # Cyan
    'Ru': (0.141, 0.561, 0.561),  # Cyan
    'Rh': (0.039, 0.49, 0.549),  # Dark Cyan
    'Pd': (0.0, 0.412, 0.522),  # Dark Cyan
    'Ag': (0.753, 0.753, 0.753),  # Light Gray
    'Cd': (1.0, 0.851, 0.561),  # Light Orange
    'In': (0.651, 0.459, 0.451),  # Muted Red
    'Sn': (0.604, 0.557, 0.725),  # Muted Violet
    'Sb': (0.62, 0.388, 0.71),  # Violet
    'Te': (0.831, 0.478, 0.0),  # Orange
    'I': (0.58, 0.0, 0.58),  # Dark Pink
    'Xe': (0.259, 0.62, 0.69),  # Cyan
    'Cs': (0.341, 0.09, 0.561),  # Dark Violet
    'Ba': (0.0, 0.788, 0.0),  # Green
    'La': (0.353, 0.769, 0.286),  # Green
    'Ce': (1.0, 1.0, 0.78),  # Very Light Yellow
    'Pr': (0.851, 1.0, 0.78),  # Very Light Green
    'Nd': (0.78, 1.0, 0.78),  # Very Light Green
    'Pm': (0.639, 1.0, 0.78),  # Light Green
    'Sm': (0.561, 1.0, 0.78),  # Light Green
    'Eu': (0.38, 1.0, 0.78),  # Light Green
    'Gd': (0.271, 1.0, 0.78),  # Green
    'Tb': (0.188, 1.0, 0.78),  # Green
    'Dy': (0.122, 1.0, 0.78),  # Cyan
    'Ho': (0.0, 1.0, 0.612),  # Green
    'Er': (0.0, 0.902, 0.459),  # Green
    'Tm': (0.0, 0.831, 0.322),  # Green
    'Yb': (0.0, 0.749, 0.22),  # Green
    'Lu': (0.0, 0.671, 0.141),  # Dark Green
    'Hf': (0.302, 0.761, 1.0),  # Blue
    'Ta': (0.302, 0.651, 1.0),  # Blue
    'W': (0.129, 0.58, 0.839),  # Blue
    'Re': (0.149, 0.49, 0.671),  # Blue
    'Os': (0.149, 0.4, 0.588),  # Blue
    'Ir': (0.09, 0.329, 0.529),  # Dark Blue
    'Pt': (0.816, 0.816, 0.878),  # Very Light Blue
    'Au': (1.0, 0.82, 0.137),  # Yellow
    'Hg': (0.722, 0.722, 0.816),  # Light Blue
    'Tl': (0.651, 0.329, 0.302),  # Red
    'Pb': (0.341, 0.349, 0.38),  # Dark Gray
    'Bi': (0.62, 0.31, 0.71),  # Violet
    'Po': (0.671, 0.361, 0.0),  # Dark Orange
    'At': (0.459, 0.31, 0.271),  # Red
    'Rn': (0.259, 0.51, 0.588),  # Cyan
    'Fr': (0.259, 0.0, 0.4),  # Very Dark Violet
    'Ra': (0.0, 0.49, 0.0),  # Dark Green
    'Ac': (0.439, 0.671, 0.98),  # Light Blue
    'Th': (0.0, 0.729, 1.0),  # Blue
    'Pa': (0.0, 0.631, 1.0),  # Blue
    'U': (0.0, 0.561, 1.0),  # Blue
    'Np': (0.0, 0.502, 1.0),  # Blue
    'Pu': (0.0, 0.42, 1.0),  # Blue
    'Am': (0.329, 0.361, 0.949),  # Blue
    'Cm': (0.471, 0.361, 0.89),  # Blue
    'Bk': (0.541, 0.31, 0.89),  # Violet
    'Cf': (0.631, 0.212, 0.831),  # Violet
    'Es': (0.702, 0.122, 0.831),  # Violet
    'Fm': (0.702, 0.122, 0.729),  # Pink
    'Md': (0.702, 0.051, 0.651),  # Pink
    'No': (0.741, 0.051, 0.529),  # Pink
    'Lr': (0.78, 0.0, 0.4),  # Pink
    'Rf': (0.8, 0.0, 0.349),  # Pink
    'Db': (0.82, 0.0, 0.31),  # Pink
    'Sg': (0.851, 0.0, 0.271),  # Pink
    'Bh': (0.878, 0.0, 0.22),  # Red
    'Hs': (0.902, 0.0, 0.18),  # Red
    'Mt': (0.922, 0.0, 0.149),  # Red
}


def make_imgs(cifs_path, output_path, rotation='30x,15y,0z', header: str = None):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)  # ensure the output folder exists

    # sorted() gives deterministic, alphabetical frame ordering (matters for make_gif later)
    pathlist = sorted(Path(cifs_path).rglob('*.cif'))
    count = 0
    for path in tqdm(pathlist, desc="Rendering structures", unit="cif"):
        atoms = read(path)
        colors = [VESTA_COLORS.get(atom.symbol, (0.5, 0.5, 0.5))
                  for atom in atoms]  # Default to gray if no color found
        fig, ax = plt.subplots()
        plot_atoms(atoms, ax, rotation=rotation, colors=colors)
        if header is None:
            plt.savefig(output_path / f"{count}.png")
        else:
            plt.savefig(output_path / f"{header}_{count}.png")
        plt.close(fig)
        count += 1


def make_gif(image_folder, output_path, duration=300):
    # Collect all PNG files and sort (optional)
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Optional, ensures correct order
    # Open images
    frames = [Image.open(os.path.join(image_folder, img))
              for img in tqdm(images, desc="Building GIF", unit="frame")]
    # Save as GIF
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,  # milliseconds per frame
        loop=0  # loop forever
    )