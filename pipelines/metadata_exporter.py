def export_metadata(dataset):
    # Export the combined metadata DataFrame to a CSV file
    dataset.to_csv("output/unified_metadata.csv", index=False)
    print("Metadata saved to unified_metadata.csv")
