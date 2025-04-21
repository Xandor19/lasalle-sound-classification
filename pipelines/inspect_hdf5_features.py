import h5py  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os
from glob import glob


def get_latest_hdf5_file(directory: str = "output") -> str:
    """
    Returns the most recently modified HDF5 file from a directory.
    """
    hdf5_files = glob(os.path.join(directory, "features_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError("No HDF5 files found in output directory.")
    return max(hdf5_files, key=os.path.getmtime)


def inspect_hdf5_file(file_path: str = None, export_txt: bool = True):
    """
    Inspect an HDF5 file created by the audio feature extraction process.

    Parameters:
    - file_path: path to the .hdf5 file to inspect. If None, the latest file is used.
    - export_txt: whether to export structure to a .txt file.
    """
    if file_path is None:
        file_path = get_latest_hdf5_file()

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    structure_lines = []

    with h5py.File(file_path, "r") as hdf:
        print("🔍 Available groups in file:")

        if len(hdf.keys()) == 0:
            print("⚠️ The file is empty or contains no groups.")
            return

        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                line = f"📁 {name}"
            elif isinstance(obj, h5py.Dataset):
                line = f"  📄 {name} -> shape: {obj.shape}"
            else:
                return
            print(line)
            structure_lines.append(line)

        hdf.visititems(print_structure)

        if export_txt:
            txt_output_path = file_path.replace(".hdf5", "_inspected.txt")
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(structure_lines))
            print(f"📄 Structure exported to: {txt_output_path}")

        try:
            first_group = next(iter(hdf.keys()))
            first_audio = next(iter(hdf[first_group].keys()))
            data = hdf[first_group][first_audio]

            if "mfcc" in data:
                mfcc = data["mfcc"][:]
                plt.figure(figsize=(10, 4))
                plt.imshow(mfcc, aspect="auto", origin="lower")
                plt.colorbar()
                plt.title(f"MFCC - {first_group}/{first_audio}")
                plt.tight_layout()
                plt.show()
            else:
                print("⚠️ No MFCC data found to plot.")
        except StopIteration:
            print("⚠️ The file does not contain valid audio groups or datasets.")


if __name__ == "__main__":
    inspect_hdf5_file()
