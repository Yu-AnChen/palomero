# Palomero Package Specification

**Version:** dynamic (managed by `setuptools-scm`) **Date:** July 4, 2025

---

## 1. Project Metadata

* **Package Name:** `palomero`
* **PyPI Name:** `palomero`
* **Author:** Yu-An Chen
* **Author Email:** <atwood12@gmail.com>
* **License:** MIT
* **Project URL:**
  <https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github>

## 2. Summary

**Palomero** is a standalone Python command-line tool for aligning two OMERO
images and transferring regions of interest (ROIs) between them. It leverages
the Palom library for a two-step alignment process: a robust, feature-based
coarse affine alignment followed by an optional non-rigid (piecewise affine)
alignment for fine-tuning. The tool is designed for both interactive single-pair
processing and automated sequential batch processing from a CSV file.

## 3. Key Features

* **Dynamic Versioning:** Version is automatically determined from Git tags
  using `setuptools-scm`.
* **Two-Step Image Alignment:** By default, performs a coarse affine alignment
  followed by a non-rigid alignment using **ITK Elastix** to accurately register
  images with minor distortions.
* **Flexible Alignment Workflow:** An `--only-affine` flag allows users to
  revert to a faster, purely affine alignment if non-rigid warping is not
  required.
* **ROI Transfer:** Maps all ROIs (and their constituent shapes) from a source
  image to a target image by applying the calculated transformation.
* **Single Pair and Batch Modes:** Process a single pair of images specified by
  their OMERO IDs, or sequentially process multiple image pairs from a
  structured CSV file.
* **Conditional OMERO Connection Handling:** A `--close-when-done` flag provides
  control over closing the OMERO connection upon successful completion.
* **Dry Run Mode:** Perform alignment and generate QC plots without posting any
  data back to the OMERO server, allowing for safe validation of alignment
  parameters.
* **Quality Control Plotting:** Automatically generates and saves JPEG images
  for visual inspection of coarse, non-rigid, and ROI mapping alignment quality.
* **Configurable Parameters:** Allows user configuration of key parameters,
  including target channels, alignment resolution (`--max-pixel-size`), and
  feature detection sensitivity (`--n-keypoints`).
* **Smart Matplotlib Backend:** Automatically selects an available `matplotlib`
  backend, ensuring compatibility across different platforms and environments.
* **Standalone CLI:** Runs as a standard Python script, authenticating via the
  user's local OMERO session store (managed by `omero login`).

## 4. Dependencies

### 4.1. Python Version

* `python >= 3.10`

### 4.2. Build-time Dependencies

* `setuptools>=61.0`
* `setuptools-scm`

### 4.3. Package Dependencies

The following packages are required for installation:

* `omero-py`
* `ezomero @
  git+https://github.com/Yu-AnChen/ezomero@9a2ab53d673e718cdf5a9dbae6ca37e9a335c815`
* `palom`
* `numpy`
* `matplotlib`
* `opencv-python`
* `tqdm`
* `itk-elastix`: For non-rigid alignment.

*(Note: The installation of `omero-py` and its complex dependency, ZeroC Ice,
can be handled by pointing pip to the Glencoe software channel.)*

## 5. Command-Line Interface (CLI) Specification

The package provides a single executable command: `palomero`.

**Usage:**

```bash
palomero (--image-id-from ID --image-id-to ID | --batch-csv FILE) [OPTIONS]
```

| Argument            | Type    | Required/Optional         | Default        | Description                                                                                   |
| :------------------ | :------ | :------------------------ | :------------- | :-------------------------------------------------------------------------------------------- |
| `--image-id-from`   | `int`   | Required (in single mode) | `None`         | OMERO ID of the source image where ROIs are mapped from.                                      |
| `--image-id-to`     | `int`   | Required (in single mode) | `None`         | OMERO ID of the target image where ROIs will be mapped to.                                    |
| `--batch-csv`       | `str`   | Required (in batch mode)  | `None`         | Path to the CSV file for batch processing. Excludes use of `--image-id-from`/`--image-id-to`. |
| `--channel-from`    | `int`   | Optional                  | `0`            | Channel index for the source image used for alignment.                                        |
| `--channel-to`      | `int`   | Optional                  | `0`            | Channel index for the target image used for alignment.                                        |
| `--max-pixel-size`  | `float` | Optional                  | `50.0`         | Selects pyramid level where pixel size is <= this value (in micrometers).                     |
| `--n-keypoints`     | `int`   | Optional                  | `10000`        | Number of features Palom should detect for coarse alignment.                                  |
| `--only-affine`     | `flag`  | Optional                  | `False`        | If set, skips the non-rigid alignment step, performing only coarse affine alignment.          |
| `--map-rois`        | `flag`  | Optional                  | `False`        | If set, performs ROI transfer after successful alignment.                                     |
| `--dry-run`         | `flag`  | Optional                  | `False`        | If set, runs alignment and generates QC plot but does not post any ROIs back to OMERO.        |
| `--qc-out-dir`      | `str`   | Optional                  | `"map-roi-qc"` | Directory where QC plot images will be saved.                                                 |
| `--close-when-done` | `flag`  | Optional                  | `False`        | If set, closes the OMERO connection only if all tasks complete successfully.                  |
| `-v`, `--verbose`   | `flag`  | Optional                  | `False`        | Enables detailed DEBUG level logging to the console.                                          |
| `-h`, `--help`      | `flag`  | Optional                  | N/A            | Shows the help message and exits.                                                             |
| `-V`, `--version`   | `flag`  | Optional                  | N/A            | Shows the dynamically generated package version and exits.                                    |

## 6. Input / Output Specification

### 6.1. Input

* **OMERO Session:** The tool requires an active OMERO session created via
  `omero login`.
* **CSV File (for Batch Mode):**
  * Must be a standard comma-separated values file with a header row.
  * The header **must** include `image_id_from` and `image_id_to`.
  * Other parameters (e.g., `channel_from`, `only_affine`) can be included as
    columns to override the command-line flags for specific rows. If not
    present, the values from the command-line flags are used.

    *Example `batch.csv`:*

    ```csv
    image_id_from,image_id_to,channel_from,channel_to
    1966810,1966783,2,0
    1966814,1966784,2,0
    ```

### 6.2. Output

* **Console (`stdout`, `stderr`):**
  * Logs progress and status updates with timestamps.
  * Prints a final summary of successful and failed tasks.
* **File System (QC Plots):**
  * Creates JPEG images in the directory specified by `--qc-out-dir`.
  * Filename format: `qc_alignment-from_<image_from>_to_<image_to>-<type>.jpg`,
    where `<type>` can be `coarse`, `elastix`, or `roi`.
* **OMERO Server:**
  * When `--map-rois` is active and `--dry-run` is not, new ROIs are created on
    the image specified by `--image_id_to`.

## 7. Project Structure

The project is structured into several files, each with a clear responsibility.

* `palomero/`
  * `.gitignore`
  * `pyproject.toml`
  * `README.md`
  * `src/`
    * `palomero/`
      * `__init__.py` (Makes `palomero` a package, defines `__version__`)
      * `models.py` (Data classes)
      * `omero_handler.py` (All direct OMERO communication logic)
      * `transform_roi_points.py` (Functions for transforming ROI points)
      * `align/` (Directory for all alignment modules)
        * `__init__.py`
        * `palom_wrapper.py` (Wraps Palom for coarse alignment)
        * `elastix_wrapper.py` (Wraps ITK Elastix for non-rigid alignment)
        * `aligner.py` (Main alignment orchestrator class)
      * `cli.py` (CLI entry point and main workflow orchestration)

### 7.1. Module Responsibilities

* **`palomero/models.py`**
  * Defines core data structures: `AlignmentTask` and `AlignmentResult`.

* **`palomero/omero_handler.py`**
  * Abstracts all interactions with the OMERO server. This is the only module
    that should make `ezomero` or `omero-py` calls.

* **`palomero/transform_roi_points.py`**
  * Provides functions for extracting and setting geometric point data for
    different ROI types.

* **`palomero/align/` (Alignment Package)**
  * **`palom_wrapper.py`**: Wraps `palom` for coarse alignment.
  * **`elastix_wrapper.py`**: Wraps `itk-elastix` for non-rigid alignment.
  * **`aligner.py`**: Orchestrates the alignment workflow, using strategies for
    affine and non-rigid alignment, and handles QC plotting.

* **`palomero/cli.py`**
  * Serves as the user-facing entry point. It is responsible for parsing
    arguments, configuring logging and matplotlib, orchestrating the high-level
    workflow (looping through tasks), and reporting results.

### 7.2. `pyproject.toml` Entry Point

The `pyproject.toml` file defines the console script entry point:

```toml
[project.scripts]
palomero = "palomero.cli:main"
```
