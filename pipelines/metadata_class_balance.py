import matplotlib.pyplot as plt  # type: ignore
import os


def analyze_class_balance(dataset, save_path="output/class_balance.png"):
    """
    Analyze and visualize the class balance (ship type distribution) in the dataset.
    Saves a bar chart as a PNG file for reporting purposes.

    Parameters:
    - dataset: pandas DataFrame with a column 'ship-type'.
    - save_path: string, file path where the PNG will be saved.
    """

    # Count how many samples exist per class (ship type)
    class_counts = dataset["ship-type"].value_counts()

    # Calculate the relative percentage of each class
    class_percent = dataset["ship-type"].value_counts(normalize=True) * 100

    # Print the absolute and percentage counts
    print("Sample count per class:\n", class_counts)
    print("\nClass percentage distribution:\n", class_percent.round(2))

    # Plot a bar chart to visualize class balance
    plt.figure(figsize=(6, 4))  # Set the figure size
    class_counts.plot(kind="bar", color="skyblue")  # Bar chart of class counts
    plt.title("Class Distribution by Ship Type")  # Chart title
    plt.xlabel("Ship Type")  # X-axis label
    plt.ylabel("Sample Count")  # Y-axis label
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add light horizontal grid lines
    plt.tight_layout()  # Prevent label cutoff

    # Save the plot as PNG in the specified path
    plt.savefig(save_path)
    print(f"\nChart saved to {os.path.abspath(save_path)}")

    # Show the chart in case it's needed interactively
    plt.show()
