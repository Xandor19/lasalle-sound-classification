import pandas as pd
import dotenv as de
import os

# Flag to prevent reloading the environment multiple times
loaded = False

def load_metadata():
    global loaded

    # Load environment variables from .env file (only once)
    if not loaded:
        de.load_dotenv()
        loaded = True

    # Get root directory of the dataset and the keyword to identify metadata files
    root = os.getenv('DATA_ROOT')
    identifier = os.getenv('METADATA_IDENTIFIER')

    if root is None or identifier is None:
        raise EnvironmentError(
            "Missing environment variables. Make sure 'DATA_ROOT' and 'METADATA_IDENTIFIER' are defined in your .env file."
        )

    if not os.path.isdir(root):
        raise FileNotFoundError(f"The directory specified in DATA_ROOT does not exist: {root}")
    
    dfs = []  # List to accumulate DataFrames from each ship class

    # Iterate over each ship class directory (e.g., Cargo, Tug, etc.)
    for ship_class in os.listdir(root):
        # Find metadata file inside the class folder based on the identifier keyword (e.g., "metafile")
        meta = [f for f in os.listdir(os.path.join(root, ship_class)) if identifier in f]

        # If a metadata file is found
        if meta:
            # Load metadata as a DataFrame with predefined column names
            df = pd.read_csv(os.path.join(root, ship_class, meta[0]),
                             names=['class-id', 'ship-name', 'day', 'hour', 'duration', 'sensor-distances', 'ship-type'],
                             index_col=0)

            # Add a column to label the type of ship (derived from folder name)
            df['ship-type'] = ship_class

            # Function to try and find the corresponding audio folder for each metadata row
            def try_get_path(row):
                base_dir = os.path.join(root, ship_class)
                try:
                    # Search for a folder that matches the day and index format (e.g., 20171104-1)
                    matching_folder = next(
                        f for f in os.listdir(base_dir)
                        if f.startswith(str(row['day'])) and f.endswith(f'-{row.name}')
                    )
                    return os.path.join(base_dir, matching_folder)
                except StopIteration:
                    return pd.NA  # Return NA if no match is found

            # Apply the audio folder search to all rows
            df['audio-dir'] = df.apply(try_get_path, axis=1)

            # Only keep entries that have a valid audio folder
            dfs.append(df.dropna(subset=['audio-dir']))

    # Combine all ship classes into a single DataFrame
    return pd.concat(dfs)
