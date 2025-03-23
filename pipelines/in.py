import pandas as pd
import dotenv as de
import os


loaded = False


def load_metadata():
    global loaded

    if not loaded:
        de.load_dotenv()
        loaded = True
    
    root = os.getenv('DATA_ROOT')
    identifier = os.getenv('METADATA_IDENTIFIER')
    dfs = []

    for ship_class in os.listdir(root):
        meta = [f for f in os.listdir(os.path.join(root, ship_class)) if identifier in f]

        if meta:
            df = pd.read_csv(os.path.join(root, ship_class, meta[0]),
                            names=['class-id', 'ship-name', 'day', 'hour', 'duration', 'sensor-distances', 'ship-type'],
                            index_col=0)
            df['ship-type'] = ship_class

            # Generate audio directory paths, leaving as NA if not found
            def try_get_path(row):
                base_dir = os.path.join(root, ship_class)
                try:
                    matching_folder = next(f for f in os.listdir(base_dir) if f.startswith(str(row['day'])) and f.endswith(f'-{row.name}'))
                    return os.path.join(base_dir, matching_folder)
                except StopIteration:
                    return pd.NA

            df['audio-dir'] = df.apply(try_get_path, axis=1)
            dfs.append(df.dropna(subset=['audio-dir']))
                
    return pd.concat(dfs)