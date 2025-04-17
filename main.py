import pandas as pd
import dotenv as de
import matplotlib.pyplot as plt
import seaborn as sns
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

def export_metadata(dataset):
    # Export the combined metadata DataFrame to a CSV file
    dataset.to_csv("output/unified_metadata.csv", index=False)
    print("Metadata saved to unified_metadata.csv")

def exploratory_data_analysis(dataset):
    """
    Perform a comprehensive exploratory data analysis (EDA) on the dataset.
    """

    # 1. Check for missing values in each column
    print("🔍 Checking for missing values:\n")
    print(dataset.isnull().sum(), "\n")

    # 2. Display data types of each column
    print("📊 Data types of each column:\n")
    print(dataset.dtypes, "\n")

    # 3. Show basic descriptive statistics for numeric columns
    print("📈 Descriptive statistics:\n")
    print(dataset.describe(), "\n")

    # 4. Plot histograms for numerical variables
    numeric_columns = dataset.select_dtypes(include='number').columns
    dataset[numeric_columns].hist(bins=15, figsize=(15, 8), layout=(2, 3), color='skyblue')
    plt.suptitle('Histogram of Numeric Features')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 5. Correlation matrix heatmap for numeric features
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.show()

    # 6. Boxplots to detect potential outliers per numeric column
    for col in numeric_columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=dataset[col], color='lightblue')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

    print("✅ EDA completed.")

def analyze_class_balance(dataset, save_path='output/class_balance.png'):
    """
    Analyze and visualize the class balance (ship type distribution) in the dataset.
    Saves a bar chart as a PNG file for reporting purposes.
    
    Parameters:
    - dataset: pandas DataFrame with a column 'ship-type'.
    - save_path: string, file path where the PNG will be saved.
    """

    # Count how many samples exist per class (ship type)
    class_counts = dataset['ship-type'].value_counts()

    # Calculate the relative percentage of each class
    class_percent = dataset['ship-type'].value_counts(normalize=True) * 100

    # Print the absolute and percentage counts
    print("Sample count per class:\n", class_counts)
    print("\nClass percentage distribution:\n", class_percent.round(2))

    # Plot a bar chart to visualize class balance
    plt.figure(figsize=(6, 4))  # Set the figure size
    class_counts.plot(kind='bar', color='skyblue')  # Bar chart of class counts
    plt.title('Class Distribution by Ship Type')  # Chart title
    plt.xlabel('Ship Type')  # X-axis label
    plt.ylabel('Sample Count')  # Y-axis label
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add light horizontal grid lines
    plt.tight_layout()  # Prevent label cutoff

    # Save the plot as PNG in the specified path
    plt.savefig(save_path)
    print(f"\nChart saved to {os.path.abspath(save_path)}")

    # Show the chart in case it's needed interactively
    plt.show()
