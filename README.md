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
  source image to a target image after alignment. Automatically handles GeoMx
  instrument ROIs (see [GeoMx ROI handling](#geomx-roi-handling)).
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

[Pixi](https://pixi.sh) is the recommended way to install Palomero. It
automatically selects the correct platform-specific dependencies (including
ZeroC Ice) for Windows, macOS, and Linux.

1. **Download the environment files** into a new directory:

    ```bash
    mkdir palomero-env && cd palomero-env

    curl -OL https://raw.githubusercontent.com/Yu-AnChen/palomero/refs/heads/main/pixi/pixi.toml
    curl -OL https://raw.githubusercontent.com/Yu-AnChen/palomero/refs/heads/main/pixi/pixi.lock
    ```

2. **Install the environment:**

    ```bash
    pixi install --locked
    ```

    > **Windows:** If git is not installed on your system, run this instead —
    > it temporarily installs git, runs the install, then removes it:
    >
    > ```bash
    > pixi global install git && pixi install --locked && pixi global remove git
    > ```

3. **Activate the environment:**

    ```bash
    pixi shell
    ```

The pixi installation includes the web app dependencies. See
[Running the Web App](#running-the-web-app) below.

## Web App (Optional GUI)

For a graphical user experience, Palomero provides a web application.

### Running the Web App

1. **Log in to OMERO:** The web app requires an active OMERO session, just like
   the CLI.

    ```bash
    omero login -s <your.omero.server> -u <username> -p <port> -t 999999
    ```

2. **Launch the app:**

    ```bash
    palomero-web
    ```

    You can then access the app in your browser at `http://localhost:5001`.

For a detailed guide on using the web app, please see the [**Web App
Tutorial**](src/palomero/web/public/TUTORIAL.md).

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
    image-id-from,image-id-to,channel-from,channel-to,only-affine
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
| `--only-affine` | Flag to skip the non-rigid alignment step. |
| `--dry-run` | Run alignment and generate QC plots but do not write ROIs to OMERO. |
| `--qc-out-dir DIR` | Specify a directory to save the QC plot images (default: `map-roi-qc`). |
| `--channel-from CH` | Channel index to use for the 'from' image (default: 0). |
| `--channel-to CH` | Channel index to use for the 'to' image (default: 0). |
| `--max-pixel-size MICRONS`| Max pixel size for selecting pyramid level (default: 20.0). |
| `--n-keypoints N` | Number of keypoints for ORB feature detection (default: 10000). |
| `--auto-mask DO_MASK` | Automatically mask out background before image alignment (default: True). |
| `--thumbnail-max-size MAX_SIZE` | Max thumbnail size when determining image orientations (default: 2000). |
| `--mask-roi-id-from ID` | ROI ID from the 'from' image to use as a mask. |
| `--mask-roi-id-to ID` | ROI ID from the 'to' image to use as a mask. |
| `--sample-size-factor S_FACTOR` | Size factor for random region sampling in elastix (default: 3.0). Recommended: 1.0-5.0. |
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

## GeoMx ROI Handling

ROIs exported from a GeoMx instrument contain multiple shapes per ROI object:

- A **geometry** (ellipse, polygon, rectangle, …) — the actual region of interest, with no text label.
- A **Label** shape — carries a zero-padded 3-digit number that identifies the ROI (e.g. `009`).
- One or more **binary mask** shapes — not supported by `ezomero`, but their text value (e.g. `Full ROI`) is still readable from the raw OMERO object.

When Palomero detects this pattern (a 3-digit Label paired with a geometry in the same ROI object), it automatically collapses all three into a single output shape: the geometry is kept and its label is set to:

```text
{3-digit} | {mask-label-1} | {mask-label-2} | …
```

For example, an ROI with digit label `009` and one mask labelled `Full ROI` produces:

```text
009 | Full ROI
```

This detection is automatic — no extra flags are needed. Non-GeoMx ROIs are processed normally.
