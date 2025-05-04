import os
import librosa
import pandas as pd
from pathlib import Path
from globals.defaults import *


def load_metadata(root, identifier, audio_extensions):
    """
    Load metadata from the specified directory and return a DataFrame with audio paths and metadata.

    Params:
    - config: Dictionary with the process configuration

    Returns:
    - DataFrame with the unified metadata and the audio files that correspond to the metadata
    """
    rows = []

    for ship_class in os.listdir(root):
        # Get the metadata file for the ship class
        meta_files = [f for f in os.listdir(os.path.join(root, ship_class)) if identifier in f]

        if meta_files:
            # It is assumed a single metadata file per ship class
            df = pd.read_csv(
                os.path.join(root, ship_class, meta_files[0]),
                names=['class-id', 'ship-name', 'day', 'hour', 'duration', 'sensor-distances', 'ship-type'],
                index_col=0
            )
            # Set the class from the parent directory
            df['ship-type'] = ship_class

            def try_get_audio_files(row):
                """
                Scans an audio directory and links all valid audio files to the metadata row

                Params:
                - row: Single metadata row with the audios directory

                Returns:

                """
                base_dir = os.path.join(root, ship_class)

                try:
                    # Finds the folder that matches the corresponding metadata record
                    folder = next(f for f in os.listdir(base_dir) if f.startswith(str(row['day'])) and f.endswith(f'-{row.name}'))
                    full_dir = Path(os.path.join(base_dir, folder))

                    if full_dir.exists() and full_dir.is_dir():
                        # Get all valid audio files in the directory 
                        audio_files = [f for f in full_dir.iterdir() if f.suffix.lower()[1:] in audio_extensions]

                        # Build a record for each audio file
                        return [(str(f.resolve()), row) for f in audio_files]
                except StopIteration:
                    return []
                return []

            for _, row in df.iterrows():
                audio_rows = try_get_audio_files(row)

                for audio_path, metadata in audio_rows:
                    # Link the audio file to the metadata
                    new_row = metadata.copy()
                    new_row['audio-path'] = audio_path
                    rows.append(new_row)

    return pd.DataFrame(rows)


def audios_from_source(root, audio_extensions,):
    audios = []
    classes = []

    for ship_class in os.listdir(root):
        for audio_dir in os.listdir(os.path.join(root, ship_class)):
            full_path = Path(os.path.join(root, ship_class, audio_dir))

            if full_path.is_dir():
                for f in full_path.iterdir():
                    if f.suffix.lower()[1:] in audio_extensions:
                        audios.append(str(f.resolve()))
                        classes.append(ship_class)

    return audios, classes


def read_audio(path, sf=None):
    """
    Reads an audio using librosa and returns the waveform vector
    """
    y, _ = librosa.load(path, sr=sf)
    return y


def tagged_audios(config, all_metadata=False, split=True):
    """
    Reads the audio files from the metadata and returns the waveform vectors and corresponding class

    Params:
    - config: Dictionary with the process configuration
    - metadata: DataFrame with the metadata. If None, it will be loaded using the load_metadata function and received config
    - split: Boolean indicating if the function should return the audio and class separately
    - all_metadata: Boolean indicating if the function should return all metadata or only the audio and class

    Returns:
    - Tuple or DataFrame with the audio, class and optionally all metadata depending on the split and all_metadata flags
    """
    root = config.get('audios-root', DATA_ROOT)
    audio_extensions = config.get('audio-extensions', AUDIO_EXTENSIONS)

    if all_metadata:
        identifier = config.get('metadata-id', METADATA_IDENTIFIER)
        metadata = load_metadata(root, identifier, audio_extensions)
        audio_dirs = metadata['audio-path']
    else:
        audio_dirs, classes = audios_from_source(root, audio_extensions)

    sf = config.get('sample-rate', SAMPLING_RATE)
    # Reads each audio
    audios = [read_audio(a, sf) for a in audio_dirs]

    if not split:
        if all_metadata:
            metadata['content'] = audios
            return metadata
        else:
            return pd.DataFrame({'content': audios, 'ship-type': classes})
    else:
        return audios, classes
