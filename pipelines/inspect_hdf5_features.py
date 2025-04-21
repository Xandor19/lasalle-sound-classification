import h5py  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os


def inspect_hdf5_file(file_path: str):
    """
    Inspect an HDF5 file created by the audio feature extraction process.

    Parameters:
    - file_path: path to the .hdf5 file to inspect.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with h5py.File(file_path, "r") as hdf:
        print("🔍 Available groups in file:")

        if len(hdf.keys()) == 0:
            print("⚠️ The file is empty or contains no groups.")
            return

        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  📄 {name} -> shape: {obj.shape}")

        hdf.visititems(print_structure)

        # Optional: plot the first MFCC found
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
                print("⚠️ No MFCC dataset found in the selected audio.")
        except StopIteration:
            print("⚠️ The file does not contain valid audio groups or datasets.")


if __name__ == "__main__":
    inspect_hdf5_file("output/features_test.hdf5")
