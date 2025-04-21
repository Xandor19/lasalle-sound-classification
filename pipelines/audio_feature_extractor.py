import os
import librosa  # type: ignore
import h5py  # type: ignore
import numpy as np  # type: ignore
import dotenv as de  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from datetime import datetime

# Load environment variables
loaded = False
if not loaded:
    de.load_dotenv()
    loaded = True

# Sampling rate (as defined in dataset)
SAMPLE_RATE = 32000


def get_audio_dataset_root() -> str:
    """
    Get the audio dataset root folder from environment variable AUDIO_ROOT.
    """
    audio_root = os.getenv("AUDIO_ROOT")
    if audio_root is None or not os.path.isdir(audio_root):
        raise EnvironmentError("AUDIO_ROOT is not defined or path is invalid.")
    return audio_root


def extract_features_from_audio(file_path: str, sr: int = SAMPLE_RATE) -> dict:
    """
    Extract audio features (MFCC, ZCR, Mel Spectrogram) from a given .wav file.

    Parameters:
    - file_path: Path to the .wav audio file.
    - sr: Target sampling rate (default is 32000 Hz).

    Returns:
    - A dictionary with keys: 'mfcc', 'zcr', 'mel_spectrogram'.
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    features = {
        "mfcc": librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13),
        "zcr": librosa.feature.zero_crossing_rate(y=audio),
        "mel_spectrogram": librosa.feature.melspectrogram(y=audio, sr=sample_rate),
    }
    return features


def process_audio_dataset(
    output_prefix: str = "output/features",
    max_files: int = None,
    save_images: bool = False,
):
    """
    Process the audio dataset and extract features for each .wav file, saving them to an HDF5 file.
    Additionally, save a text file listing the HDF5 structure and PNG images of MFCCs (if enabled).

    Parameters:
    - output_prefix: Prefix for output files (without extension).
    - max_files: Optional maximum number of audio files to process (for testing).
    - save_images: Whether to save MFCC images as PNG.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_hdf5 = f"{output_prefix}_{timestamp}.hdf5"
    output_txt = f"{output_prefix}_{timestamp}_structure.txt"
    mfcc_img_folder = os.path.join("output", f"mfcc_png_{timestamp}")

    root = get_audio_dataset_root()
    processed_files = 0
    structure_lines = []

    with h5py.File(output_hdf5, "w") as hdf:
        for category in os.listdir(root):
            category_path = os.path.join(root, category)
            if not os.path.isdir(category_path):
                continue

            print(f"Processing category: {category}")
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                for audio_file in os.listdir(subfolder_path):
                    if not audio_file.endswith(".wav"):
                        continue

                    audio_path = os.path.join(subfolder_path, audio_file)
                    group_name = (
                        f"{category}/{subfolder}/{audio_file.replace('.wav','')}"
                    )

                    try:
                        features = extract_features_from_audio(audio_path)
                        group = hdf.create_group(group_name)
                        for feature_name, data in features.items():
                            group.create_dataset(feature_name, data=data)
                        print(f"  ✔ Processed: {group_name}")

                        if save_images and "mfcc" in features:
                            plt.figure(figsize=(8, 3))
                            plt.imshow(features["mfcc"], aspect="auto", origin="lower")
                            plt.title(f"MFCC - {audio_file}")
                            plt.tight_layout()
                            os.makedirs(mfcc_img_folder, exist_ok=True)
                            image_path = os.path.join(
                                mfcc_img_folder, f"{audio_file.replace('.wav','.png')}"
                            )
                            plt.savefig(image_path)
                            plt.close()

                        structure_lines.append(f"{group_name}")

                    except Exception as e:
                        print(f"  ✖ Error processing {audio_path}: {e}")

                    processed_files += 1
                    if max_files is not None and processed_files >= int(max_files):
                        print("⚠️ Max files limit reached. Stopping early.")
                        break

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(structure_lines))

    print(f"✅ Structure exported to: {output_txt}")
    print(f"✅ Features saved in: {output_hdf5}")
    print(f"✅ Finished processing. Total files: {processed_files}")


if __name__ == "__main__":
    process_audio_dataset()
