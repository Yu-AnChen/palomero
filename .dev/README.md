# Map OMERO ROI (Standalone CLI Tool)

## Purpose

This script aligns two OMERO images and optionally transfers Region of Interests
(ROIs) from the second image (e.g., H&E) to the first image (e.g., IF).

The alignment process involves:

1. Fetching specified channels from potentially downsampled pyramid levels of
    the two images (selected based on `--max-pixel-size`).
2. Performing coarse affine registration between the two image channels using
    the [palom](https://github.com/labsyspharm/palom) library (using ORB
    features by default).
3. Generating a Quality Control (QC) plot showing the detected and matched
    features.

If ROI mapping is requested (`--map-rois` flag):

1. The script calculates the affine transformation.
2. It fetches all shapes associated with ROIs on the second image.
3. It applies the calculated transformation to these shapes.
4. It uploads the transformed shapes as new ROIs (one ROI per shape) onto the
   first image.

The script is provided as a standalone Command Line Interface (CLI) tool. It can
process a single pair of OMERO images specified by their IDs or run in batch
mode, processing multiple pairs sequentially from a provided CSV file.

## Requirements

* **Python:** 3.8+ (Tested with 3.10)
* **OMERO Python Bindings:** `omero-py`
* **ZeroC Ice:** Specific version compatible with your Python and OS (see
  installation).
* **Alignment:** `palom` (which often requires `opencv-python`)
* **ROI Handling:** `ezomero` (specific commit required)
* **Plotting (Optional but Recommended):** `matplotlib` (for QC plots)

## Installation

1. **Conda Environment (Recommended):** Using Miniconda or Anaconda is
    recommended for managing dependencies.

    ```bash
    # Create a Conda environment (e.g., named "maproi_env")
    # Adjust python version if needed
    conda create -n maproi_env -c conda-forge python=3.10 pip

    # Activate the environment
    conda activate maproi_env
    ```

2. **Install ZeroC Ice:**

    `omero-py` depends on ZeroC Ice. Download the correct wheel for your OS and
    Python version from [Glencoe software's
    GitHub](https://github.com/glencoesoftware/). Search for "zeroc" for the
    repos.

    *Example for Windows Python 3.10:*

    ```bash
    # Download using curl (often available on Win10+) or wget or manually
    curl -L -O https://github.com/glencoesoftware/zeroc-ice-py-wheels/releases/download/v3.6.5.1/zeroc_ice-3.6.5.1-cp310-cp310-win_amd64.whl
    # Install the wheel
    python -m pip install zeroc_ice-3.6.5.1-cp310-cp310-win_amd64.whl
    # Clean up downloaded file
    del zeroc_ice-3.6.5.1-cp310-cp310-win_amd64.whl
    ```

    *Example for Linux Python 3.10:*

    ```bash
    curl -L -O https://github.com/glencoesoftware/zeroc-ice-py-wheels/releases/download/v3.6.5.1/zeroc_ice-3.6.5.1-cp310-cp310-manylinux2014_x86_64.whl
    python -m pip install zeroc_ice-3.6.5.1-cp310-cp310-manylinux2014_x86_64.whl
    rm zeroc_ice-3.6.5.1-cp310-cp310-manylinux2014_x86_64.whl
    ```

    *Example for macOS (Apple Silicon) Python 3.10:*

    ```bash
    curl -L -O https://github.com/glencoesoftware/zeroc-ice-py-macos-universal2/releases/download/20240131/zeroc_ice-3.6.5-cp310-cp310-macosx_11_0_universal2.whl
    python -m pip install zeroc_ice-3.6.5-cp310-cp310-macosx_11_0_universal2.whl
    rm zeroc_ice-3.6.5-cp310-cp310-macosx_11_0_universal2.whl
    ```

    *(Adapt filename for other OS or  Python versions based on the releases
    page)*

3. **Install Python Dependencies:**

    ```bash
    # Install omero-py, palom
    python -m pip install omero-py palom

    # Install ezomero from the specific required commit
    python -m pip install 'ezomero @ git+https://github.com/Yu-AnChen/ezomero@9a2ab53d673e718cdf5a9dbae6ca37e9a335c815'
    
    ```

4. **Get the Script:**

    [Download the script
    file](https://github.com/Yu-AnChen/dump/blob/main/2024-10/map-roi/map-roi.py)
    (`map-roi.py`) and place it in your desired working directory or a location
    included in your system's PATH.

## Usage

### OMERO Login

The script uses your locally stored OMERO session credentials. You need to log
in first using the OMERO CLI:

```bash
# Log in to your OMERO server (replace placeholders)
# The -t flag sets session timeout in seconds (optional but recommended for long runs)
omero login -s <your.omero.server> -u <username> -p <port> -t 999999
```

Enter your password when prompted.

### Running the Script (Pair Mode)

* Single Image Pair (Dry Run - Alignment Check): Aligns image `<SECOND-ID>` to
`<FIRST-ID>`, saves QC plot, but does not transfer ROIs.

    ```bash
    python map-roi.py --image-id-1 <FIRST-ID> --image-id-2 <SECOND-ID> --qc-out-dir path/to/qc_plots --dry-run
    ```

    Check the generated `.jpg` file in `path/to/qc_plots`. Successful alignment
    is typically indicated by blue lines connecting corresponding features
    across the two image overlays.

* Single Image Pair (with ROI Mapping):

    Performs alignment and QC plot generation, then transfers ROIs from
    `<SECOND-ID>` to `<FIRST-ID>`.

    ```bash
    python map-roi.py --image-id-1 <FIRST-ID> --image-id-2 <SECOND-ID> --map-rois
    ```

    *(QC plots go to the default map-roi-qc directory unless `--qc-out-dir` is
    specified)*

* Using Specific Channels:

    Align using channel index 1 (second channel, 0-based indexing) for both
    images.

    ```bash
    python map-roi.py --image-id-1 <FIRST-ID> --image-id-2 <SECOND-ID> --channel-1 1 --channel-2 1 --map-rois
    ```

* Adjusting Alignment Parameters:

    Use a different maximum pixel size for selecting alignment resolution and
    change the number of keypoints.

    ```bash
    python map-roi.py --image-id-1 <FIRST-ID> --image-id-2 <SECOND-ID> --max-pixel-size 100 --n-keypoints 5000 --map-rois
    ```

* Verbose Output:

    Get more detailed DEBUG level logging during execution.

    ```bash
    python map-roi.py --image-id-1 <FIRST-ID> --image-id-2 <SECOND-ID> --map-rois -v
    ```

### Batch Mode

Process multiple image pairs sequentially from a CSV file.

* CSV File Format:

    The CSV file must contain a header row with at least these column names:
    `image_id_1`, `image_id_2`. Options available in pair mode can also be used
    in batch mode, either as additional columns in the CSV file or as flags in
    the command line call. If both are provided, the values in the CSV file take
    precedence.

    *Example (batch_file.csv)*:

    ```CSV
    image_id_1,image_id_2,channel_1,channel_2
    1614258,1623948,1,1
    1614258,1623951,1,1
    1614701,1623954,
    1614701,1623957,1,1
    ```

* Running Batch Mode:

    ```bash
    # Run alignment and ROI mapping for all pairs in the CSV
    python map-roi.py --batch-csv path/to/batch_file.csv --map-rois --qc-out-dir path/to/batch_qc

    # Run batch mode as a dry run (only alignment check and QC plots)

    python map-roi.py --batch-csv path/to/batch_file.csv --dry-run --qc-out-dir path/to/batch_qc_dry
    ```

### Command-Line Arguments Summary

```bash
usage: map-roi.py [-h] (--batch-csv FILE | --image-id-1 ID) [--image-id-2 ID]
                  [--channel-1 CH] [--channel-2 CH] [--max-pixel-size MICRONS]
                  [--n-keypoints N] [--auto-mask DO_MASK]
                  [--thumbnail-max-size MAX_SIZE] [--map-rois] [--dry-run]
                  [--qc-out-dir DIR] [-v] [--secure SECURE]
                  [--keepalive KEEPALIVE]

Align two OMERO images or a batch from CSV (sequentially), generate QC plots,
and optionally map ROIs.

options:
  -h, --help            show this help message and exit
  --batch-csv FILE      Path to CSV file for batch processing (processed
                        sequentially). Required columns: image_id_1,
                        image_id_2, channel_1, channel_2. (default: None)
  --image-id-1 ID       Omero Image ID for the first image (e.g.,
                        reference/IF). (default: None)
  --image-id-2 ID       Omero Image ID for the second image (e.g., target/HE).
                        Required if --image-id-1 is provided. (default: None)
  --channel-1 CH        Default channel index for the first image. Default: 0
  --channel-2 CH        Default channel index for the second image. Default: 0
  --max-pixel-size MICRONS
                        Maximum pixel size for selecting pyramid level for
                        alignment. Default: 50.0 µm
  --n-keypoints N       Number of keypoints for Palom SIFT feature detection.
                        Default: 10000
  --auto-mask DO_MASK   Automatically mask out background before image
                        alignment. Default: True
  --thumbnail-max-size MAX_SIZE
                        Max thumbnail size when determining image
                        orientations. Default: 2000
  --map-rois            Attempt to map ROIs from image_id_2 to image_id_1.
                        Ignored if --dry-run is set. (default: False)
  --dry-run             Perform alignment and QC plot generation, but DO NOT
                        map/post ROIs. (default: False)
  --qc-out-dir DIR      Output directory for Quality Control (QC) alignment
                        plots. Default: map-roi-qc
  -v, --verbose         Enable verbose (DEBUG level) logging. (default: False)
  --secure SECURE       Use secure session. Default: True
  --keepalive KEEPALIVE
                        Do not logout before exiting. Default: True
```

## Output

* **Console**: Logs progress, informational messages, warnings, and errors. A
  final summary indicates successful and failed tasks.

* **QC Plots**: JPEG images showing the alignment overlay are saved in the
  directory specified by `--qc-out-dir` (default: `map-roi-qc`). Filenames
  include the image IDs.

* **OMERO**: If `--map-rois` is used (and not `--dry-run`), new ROIs
  corresponding to the transformed shapes from the second image will be created
  on the first image.

## Troubleshooting

* **Connection Errors**: Ensure you have run omero login successfully and your
  session hasn't expired. Check server address and credentials. You may need to
  use `--secure=False` in the map-roi.py call depending on your omero server
  configuration

* **Alignment Failures**: Poor alignment can occur with images that have very
  different content, staining, or significant non-affine distortions. Check the
  QC plot. Adjusting `--n-keypoints`, `--channel-1/2`, `--max-pixel-size` or
  other  parameters within the script might be necessary for difficult cases.

* **Import Errors**: Double-check that all dependencies listed in the
  Installation section are installed correctly in the active Python environment
  (conda activate maproi_env). Pay special attention to the ZeroC Ice wheel
  version compatibility.

* **Permissions Errors**: The script needs read access to both images and write
  access (for ROIs) to the first image if `--map-rois` is used. Ensure your
  OMERO user has the necessary permissions.