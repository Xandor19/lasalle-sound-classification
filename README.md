# lasalle-sound-classification

Final Project of the La Salle Big Data Master Data Mining subject

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Description

This project aims to classify sounds using data processing techniques and machine learning. The dataset is stored in the `resources/data` folder, and metadata is defined in the `.env` file for data identification and processing.

## Requirements

Make sure you have the following components installed:

- Python 3.8 or higher
- Libraries listed in `requirements.txt`

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/tu-usuario/lasalle-sound-classification.git
   cd lasalle-sound-classification
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables in the .env file:
   ```bash
   DATA_ROOT="resources/data"
   METADATA_IDENTIFIER="metafile"
   ```

## Usage

1.Ensure the dataset is located in the folder specified by DATA_ROOT.
2.Run the main script to start the classification process:

```bash
    python main.py
```

3. The results will be saved in the output folder (or the path configured in the project).

## Estructura del Proyecto

```bash
lasalle-sound-classification/
│
├── resources/
│ └── data/ # Input data
├── output/ # Generated results
├── src/ # Source code
│ ├── preprocessing/ # Preprocessing scripts
│ ├── models/ # Classification models
│ └── utils/ # Utility functions
├── .env # Environment variables
├── [requirements.txt](http://_vscodecontentref_/1) # Project dependencies
├── [README.md](http://_vscodecontentref_/2) # Project documentation
└── main.py # Main script
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to suggest improvements.

## License

This project is part of an academic exercise for the La Salle Big Data Master's program. It is intended for educational purposes only and is not meant for commercial use. If you wish to use or adapt this project, please contact the author or institution for permission.
