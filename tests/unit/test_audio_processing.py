from pipelines.audio_feature_extractor import process_audio_dataset


# Run feature extraction for a limited number of files (for testing)
def test_extract_audio_features():
    process_audio_dataset(output_path="output/features_test_5.hdf5", max_files=5)


if __name__ == "__main__":
    test_extract_audio_features()
