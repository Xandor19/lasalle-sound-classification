import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore


def exploratory_analysis(dataset):
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
    """ print("📈 Descriptive statistics:\n")
    print(dataset.describe(), "\n") """

    # 4. Plot histograms for numerical variables
    numeric_columns = dataset.select_dtypes(include="number").columns
    dataset[numeric_columns].hist(
        bins=15, figsize=(15, 8), layout=(2, 3), color="skyblue"
    )
    plt.suptitle("Histogram of Numeric Features")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 5. Correlation matrix heatmap for numeric features
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        dataset[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Correlation Matrix of Numeric Features")
    plt.tight_layout()
    plt.show()

    # 6. Boxplots to detect potential outliers per numeric column
    for col in numeric_columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=dataset[col], color="lightblue")
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

    print("✅ EDA completed.")
