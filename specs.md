# Palomero Package Specification

**Version:** 0.2.0 **Date:** June 30, 2025

---

## 1. Project Metadata

* **Package Name:** `palomero`
* **PyPI Name:** `palomero` (pending availability check)
* **Author:** [Your Name/Organization]
* **Author Email:** [your.email@example.com]
* **License:** MIT (or your preferred open-source license)
* **Project URL:**
  <https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github>

## 2. Summary

**Palomero** is a standalone Python command-line tool for aligning two OMERO
images and transferring regions of interest (ROIs) between them. It leverages
the Palom library for a two-step alignment process: a robust, feature-based
coarse affine alignment followed by a non-rigid (piecewise affine) alignment for
fine-tuning. The tool is designed for both interactive single-pair processing
and automated sequential batch processing from a CSV file.

## 3. Key Features

* **Two-Step Image Alignment:** By default, performs a coarse affine alignment
  followed by a non-rigid alignment using **ITK Elastix** to accurately register
  images with minor distortions.
* **Flexible Alignment Workflow:** An `--affine-only` flag allows users to
  revert to a faster, purely affine alignment if non-rigid warping is not
  required.
* **ROI Transfer:** Maps all ROIs (and their constituent shapes) from a source
  image to a target image by applying the calculated transformation.
* **Single Pair Mode:** Process a single pair of images specified directly by
  their OMERO IDs.
* **Batch Mode:** Sequentially process multiple image pairs from a structured
  CSV file.
* **Dry Run Mode:** Perform alignment and generate QC plots without posting any
  data back to the OMERO server, allowing for safe validation of alignment
  parameters.
* **Quality Control Plotting:** Automatically generates and saves a JPEG image
  overlaying the two aligned images for visual inspection of alignment quality.
* **Configurable Parameters:** Allows user configuration of key parameters,
  including target channels, alignment resolution (`--max-pixel-size`), and
  feature detection sensitivity (`--n-keypoints`).
* **Standalone CLI:** Runs as a standard Python script, authenticating via the
  user's local OMERO session store (managed by `omero login`).

## 4. Dependencies

### 4.1. Python Version

* `python >= 3.10`

### 4.2. Package Dependencies

The following packages will be listed as `install_requires`:

* `omero-py`
* `ezomero @
  git+https://github.com/Yu-AnChen/ezomero@9a2ab53d673e718cdf5a9dbae6ca37e9a335c815`
* `palom`
* `numpy`
* `matplotlib`
* `opencv-python`
* `tqdm`
* `itk-elastix`: For non-rigid alignment.

*(Note: ZeroC Ice is a runtime dependency of `omero-py` but is typically
installed manually by the user via a platform-specific wheel.)*

## 5. Command-Line Interface (CLI) Specification

The package will provide a single executable command: `palomero`.

**Usage:**

```bash
palomero (--image-from ID --image-to ID | --batch-csv FILE) [OPTIONS]
```

| Argument           | Type    | Required/Optional         | Default        | Description                                                                                 |
| :----------------- | :------ | :------------------------ | :------------- | :------------------------------------------------------------------------------------------ |
| `--image-from`     | `int`   | Required (in single mode) | `None`         | OMERO ID of the source image where ROIs are mapped from.                                    |
| `--image-to`       | `int`   | Required (in single mode) | `None`         | OMERO ID of the target image where ROIs will be mapped to.                                  |
| `--batch-csv`      | `str`   | Required (in batch mode)  | `None`         | Path to the CSV file for batch processing. Excludes use of `--image-from`/`--image-to`.     |
| `--channel-from`   | `int`   | Optional                  | `0`            | Channel index for the source image used for alignment.                                      |
| `--channel-to`     | `int`   | Optional                  | `0`            | Channel index for the target image used for alignment.                                      |
| `--max-pixel-size` | `float` | Optional                  | `50.0`         | Selects pyramid level where pixel size is <= this value (in micrometers).                   |
| `--n-keypoints`    | `int`   | Optional                  | `10000`        | Number of features Palom should detect for coarse alignment.                                |
| `--affine-only`    | `flag`  | Optional                  | `False`        | If set, skips the non-rigid alignment step, performing only coarse affine alignment.        |
| `--map-rois`       | `flag`  | Optional                  | `False`        | If set, performs ROI transfer after successful alignment. Ignored if `--dry-run` is active. |
| `--dry-run`        | `flag`  | Optional                  | `False`        | If set, runs alignment and generates QC plot but does not post any ROIs back to OMERO.      |
| `--qc-out-dir`     | `str`   | Optional                  | `"map-roi-qc"` | Directory where QC plot images will be saved.                                               |
| `-v`, `--verbose`  | `flag`  | Optional                  | `False`        | Enables detailed DEBUG level logging to the console.                                        |
| `-h`, `--help`     | `flag`  | Optional                  | N/A            | Shows the help message and exits.                                                           |
| `-V`, `--version`  | `flag`  | Optional                  | N/A            | Shows the package version and exits.                                                        |

## 6. Input / Output Specification

### 6.1. Input

* **OMERO Session:** The tool requires an active OMERO session created via
  `omero login`.
* **CSV File (for Batch Mode):**
  * Must be a standard comma-separated values file with a header row.
  * The header **must** include the following columns: `image_from`, `image_to`,
    `channel_from`, `channel_to`.

    *Example `batch.csv`:*

    ```csv
    image_from,image_to,channel_from,channel_to
    1623948,1614258,1,1
    1623951,1614258,1,1
    ```

### 6.2. Output

* **Console (`stdout`, `stderr`):**
  * Logs progress and status updates.
  * Prints a final summary of successful and failed tasks.
* **File System (QC Plots):**
  * Creates JPEG images in the directory specified by `--qc-out-dir`.
  * Filename format: `qc_alignment_<image_from>_to_<image_to>.jpg`.
* **OMERO Server:**
  * When `--map-rois` is active, new ROIs are created on the image specified by
    `--image-to`.

## 7. Recommended Project Structure

To ensure the codebase is modular, testable, and easy to maintain, the project
will be structured into several files, each with a clear responsibility.

* `palomero/`
  * `.gitignore`
  * `pyproject.toml` (or `setup.py`)
  * `README.md`
  * `src/`
    * `palomero/`
      * `__init__.py` (Makes `palomero` a package)
      * `models.py` (Data classes)
      * `omero_handler.py` (All direct OMERO communication logic)
      * `align/` (Directory for all alignment modules)
        * `__init__.py`
        * `palom_wrapper.py` (Wraps Palom for coarse alignment)
        * `elastix_wrapper.py` (Wraps ITK Elastix for non-rigid alignment)
        * `aligner.py` (Main alignment orchestrator class)
      * `cli.py` (CLI entry point and main workflow orchestration)

### 7.1. Module Responsibilities

* **`palomero/models.py`**
  * **Purpose:** To define the core data structures used throughout the
    application. This decouples the logic from raw dictionaries.
  * **Contents:**
    * `AlignmentTask`: Dataclass holding all parameters for a single alignment
      job.
    * `AlignmentResult`: Dataclass for returning structured results from an
      alignment job.

* **`palomero/omero_handler.py`**
  * **Purpose:** To abstract all interactions with the OMERO server. This module
    is the only place where `ezomero` or direct `omero-py` calls should be made.
  * **Contents:**
    * `ImageHandler` class: Fetches and caches image metadata.
    * `get_omero_connection()`: Establishes a connection to OMERO via the
      session store.
    * `fetch_rois()`: Fetches shapes from a source image.
    * `transform_rois()`: Applies a given transformation to a list of shapes.
    * `post_rois()`: Posts a list of transformed shapes to a target image.

* **`palomero/align/` (Alignment Package)**

  * **`palomero/align/palom_wrapper.py`**
    * **Purpose:** To specifically wrap `palom` library functions for coarse
      alignment.
    * **Contents:**
      * `PalomReaderFactory` class: Creates `palom`-compatible image readers
        from an `ImageHandler` object.
      * `run_coarse_alignment()`: Takes two `PalomReader` objects and returns
        the coarse affine transformation matrix and the `palom.Aligner` object.

  * **`palomero/align/elastix_wrapper.py`**
    * **Purpose:** To contain all logic for performing non-rigid alignment using
      `itk-elastix`.
    * **Contents:**
      * `run_non_rigid_alignment()`: Takes the two images (as ITK images) and
        the initial affine transform, runs Elastix, and returns the final
        non-rigid transformation object.

  * **`palomero/align/aligner.py`**
    * **Purpose:** To orchestrate the entire alignment workflow.
    * **Contents:**
      * `OmeroRoiAligner` class:
        * `__init__()`: Takes an `AlignmentTask` dataclass.
        * `execute()`: A high-level method that runs the full pipeline: 1. Calls
                    `palom_wrapper.run_coarse_alignment()`. 2. If `affine_only`
                    is false, it calls
                    `elastix_wrapper.run_non_rigid_alignment()`. 3. Returns the
                    final transformation object (affine or non-rigid).
      * `generate_qc_plot()`: Takes the `palom.Aligner` object (and potentially
        non-rigid results) to produce and save the QC image.

* **`palomero/cli.py`**
  * **Purpose:** To serve as the user-facing entry point. It is responsible for
    parsing arguments, orchestrating the high-level workflow, and reporting
    results.
  * **Contents:**
    * `main()`: The primary function called by the console script entry point.
    * `argparse` setup and validation.
    * **Workflow Orchestration:** 1. Parse arguments. 2. Call
            `omero_handler.get_omero_connection()`. 3. Prepare a list of
            `AlignmentTask` objects (from args or CSV). 4. Loop through each
            `task`: a.  Instantiate the `OmeroRoiAligner(task)`. b.  Call the
            aligner's `execute()` method to get the final transformation. c.
                Generate the QC plot. d.  If not `--dry-run`, use
                `omero_handler` functions to fetch, transform, and post the ROIs
                using the final transformation. 5. Report the final summary.

### 7.2. `pyproject.toml` Entry Point

The `pyproject.toml` file will define the console script entry point:

```toml
[project.scripts]
palomero = "palomero.cli:main"
