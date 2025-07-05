# Palomero

Palomero is a command-line tool for aligning whole-slide images stored in an
OMERO server and transferring regions of interest (ROIs) between them.

It uses a robust, two-step alignment process:

1. **Coarse Alignment**: A fast, feature-based affine alignment using the
    `palom` library.
2. **Non-Rigid Alignment**: A fine-tuning step using `itk-elastix` to correct
    for non-linear distortions. This step is optional and can be skipped for
    faster processing.

## Key Features

- **Single Pair & Batch Processing**: Align a single pair of images using their
  OMERO IDs, or process many pairs in sequence from a CSV file.
- **ROI Transfer**: Maps all ROI types (polygons, points, lines, etc.) from a
  source image to a target image after alignment.
- **Quality Control (QC) Plots**: Automatically generates JPEG images to
  visually inspect the quality of the alignment at each stage (coarse,
  non-rigid, and ROI mapping).
- **Dry Run Mode**: Run the entire alignment process and generate QC plots
  without writing any data back to the OMERO server, perfect for testing
  parameters.
- **Flexible Configuration**: Control the alignment process with various
  parameters, including target channels, alignment resolution, and feature
  detection settings.

## Installation

1. **Conda Environment (Recommended):** Using Miniconda or Anaconda is
    recommended for managing dependencies.

    ```bash
    # Create a Conda environment (e.g., named "palomero_env")
    conda create -n palomero_env -c conda-forge python=3.10 pip

    # Activate the environment
    conda activate palomero_env
    ```

2. **Install ZeroC Ice:**

    `omero-py` depends on ZeroC Ice. Use pip to install the the correct wheel
    for your OS and Python version from [Glencoe software's
    GitHub](https://github.com/glencoesoftware/). Search for "zeroc" for the
    repos.

    *Example for Windows Python 3.10:*

    ```bash
    python -m pip install https://github.com/glencoesoftware/zeroc-ice-py-win-x86_64/releases/download/20240325/zeroc_ice-3.6.5-cp310-cp310-win_amd64.whl
    ```

    *Example for macOS Python 3.10:*

    ```bash
    python -m pip install https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases/download/20240131/zeroc_ice-3.6.5-cp310-cp310-macosx_11_0_universal2.whl
    ```

    *Example for Linux Python 3.10:*

    ```bash
    python -m pip install https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp310-cp310-manylinux_2_28_x86_64.whl
    ```

3. **Install palomero:**

    ```bash
    python -m pip install "palomero @ git+https://github.com/yu-anchen/palomero@main"
    ```

## Usage

### 1. Log in to OMERO

The script uses your locally stored OMERO session credentials. You need to log
in first using the [OMERO
CLI](https://omero.readthedocs.io/en/stable/users/cli/index.html):

```bash
# Log in to your OMERO server (replace placeholders)
# The -t flag sets session timeout in seconds (optional but recommended for long runs)
omero login -s <your.omero.server> -u <username> -p <port> -t 999999
```

Enter your password when prompted.

### 2. Run Palomero

The tool can be run in two main modes: for a single pair of images or for a
batch of pairs defined in a CSV file.

#### **Single Pair Example**

Align two images and map the ROIs from the 'from' image to the 'to' image.

```bash
palomero \
    --image-id-from 101 \
    --image-id-to 202 \
    --channel-from 1 \
    --channel-to 0 \
    --map-rois
```

#### **Batch Mode**

Process multiple image pairs sequentially from a CSV file.

- CSV File Format:

    The CSV file must contain a header row with at least these column names:
    `image-id-from`, `image-id-to`. Options available in pair mode can also be
    used in batch mode, either as additional columns in the CSV file (without
    the leading `--`) or as flags in the command line call. If both are
    provided, the values in the CSV file take precedence.

    *Example (batch_file.csv)*:

    ```csv
    image-id-from,image-id-to,channel-from,channel-to,affine-only
    101,202,1,0,False
    103,204,2,1,True
    105,206,1,0,False
    ```

- Running Batch Mode:

    ```bash
    # Run alignment and ROI mapping for all pairs in the CSV
    palomero --batch-csv path/to/batch_file.csv --map-rois --qc-out-dir path/to/batch_qc

    # Run batch mode as a dry run (only alignment check and QC plots)

    palomero --batch-csv path/to/batch_file.csv --dry-run --qc-out-dir path/to/batch_qc_dry
    ```

### Command-Line Arguments

Here are some of the most important arguments. For a full list, run `palomero
--help`.

| Argument | Description |
| :--- | :--- |
| `--image-id-from ID` | OMERO ID of the source image (where ROIs are). |
| `--image-id-to ID` | OMERO ID of the target image (where ROIs will be mapped). |
| `--batch-csv FILE` | Path to a CSV file for batch processing. |
| `--map-rois` | Flag to enable the transfer of ROIs after alignment. |
| `--affine-only` | Flag to skip the non-rigid alignment step. |
| `--dry-run` | Run alignment and generate QC plots but do not write ROIs to OMERO. |
| `--qc-out-dir DIR` | Specify a directory to save the QC plot images (default: `map-roi-qc`). |
| `--channel-from CH` | Channel index to use for the 'from' image (default: 0). |
| `--channel-to CH` | Channel index to use for the 'to' image (default: 0). |
| `--max-pixel-size F`| The desired resolution (in microns per pixel) for the alignment (default: 50.0). |
| `--n-keypoints N` | Number of features to detect for coarse alignment (default: 10000). |
| `--close-when-done` | Close the OMERO connection upon successful completion of all tasks. |
| `-v`, `--verbose` | Enable detailed logging for debugging. |
| `-V`, `--version` | Show the version number. |

### Output

- **Console**: Logs progress, informational messages, warnings, and errors. A
  final summary indicates successful and failed tasks.

- **QC Plots**: JPEG images showing the alignment overlay are saved in the
  directory specified by `--qc-out-dir` (default: `map-roi-qc`). Filenames
  include the image IDs.

- **OMERO**: If `--map-rois` is used (and not `--dry-run`), new ROIs
  corresponding to the transformed shapes from the second image will be created
  on 'to' image.
