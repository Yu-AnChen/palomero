# Alignment Tool Comparison

**`palom-align-he`** (from the `palom` package) is designed for aligning
whole-slide images acquired from **the same tissue section** with different
stains or on different microscopes/magnifications. **`LocalPalomeroAligner`**
(from this repo) shares the same coarse alignment engine but uses itk-elastix
for refinement, making it applicable beyond the same-section constraint.

## When to use which

| | `palom-align-he` | `LocalPalomeroAligner` |
| :--- | :--- | :--- |
| **Works for same-section pairs** | Yes — and excels here | Yes, but typically less accurate than palom |
| **Works beyond same-section** | No — phase correlation breaks down | Yes |
| **Refinement** | Block-shift (phase correlation) | Non-rigid (itk-elastix) |
| **Output** | Warped OME-TIFF | Warped OME-TIFF |
| **Interface** | CLI (`palom-align-he`) | Python class |

**Pick `palom-align-he` when** the two images come from the same tissue section.
Its block-shift phase-correlation refinement is well-suited here because
corresponding regions stay close after coarse alignment — it is both accurate
and straightforward to run from the command line.

**Pick `LocalPalomeroAligner` when** the image pair goes beyond the same-section
assumption (e.g., serial sections with larger morphological differences). It
uses itk-elastix, which is more flexible but tends to be less precise than
phase correlation for true same-section pairs.

## Recommended workflow

For either tool, it is worth taking a staged approach before committing to a
full image warp:

**`palom-align-he`**

1. Run coarse alignment only and inspect the QC plots:

   ```bash
   palom-align-he run-pair P1.ome.tiff P2.vsi /path/to/out_dir --only_coarse --only_qc
   ```

2. If coarse alignment looks good, run the full pipeline (refined alignment +
   warp) — the refinement step is fast enough that a separate QC check is
   usually not necessary:

   ```bash
   palom-align-he run-pair P1.ome.tiff P2.vsi /path/to/out_dir
   ```

**`LocalPalomeroAligner`**

1. Run affine-only with `dry_run=True` and inspect the QC plots.
2. If affine looks good, run with elastix refinement but still `dry_run=True`
   to inspect the non-rigid QC plots before committing.
3. Once satisfied, run with `dry_run=False` to write the final warped OME-TIFF.

## How to use `palom-align-he`

`palom-align-he` is included with the palomero installation (palomero depends on palom).

```bash
# Align a single pair
palom-align-he run-pair P1.ome.tiff P2.vsi /path/to/out_dir

# Quick check: coarse alignment + QC plots only, no warp
palom-align-he run-pair P1.ome.tiff P2.vsi /path/to/out_dir \
    --only_coarse --only_qc

# Batch mode from a CSV (columns: p1, p2, and any run-pair flag)
palom-align-he run-batch pairs.csv --out_dir /path/to/out_dir
```

Key flags:

| Flag | Default | Notes |
| :--- | :--- | :--- |
| `--thumbnail_channel1` | `1` | IF channel used for coarse alignment |
| `--channel1` / `--channel2` | `0` / `2` | Channels for refined alignment |
| `--n_keypoints` | `10000` | ORB keypoints for coarse stage |
| `--only_coarse` | `False` | Skip refined block-shift step |
| `--only_qc` | `False` | Skip warping the moving image |
| `--auto_mask` | `True` | Mask non-tissue background |

See [`palom-align-he` user guide](https://github.com/labsyspharm/palom/blob/main/palom/cli/doc/align_he.md)
for full documentation and troubleshooting tips.

## How to use `LocalPalomeroAligner`

Install palomero (see [README](README.md)).

```python
from palomero.local_aligner import LocalPalomeroAligner
from palomero.models import AlignmentTask

task = AlignmentTask(
    image_id_from=0,          # unused for local files; set to 0
    image_id_to=0,
    channel_from=0,           # channel index in the reference image
    channel_to=0,             # channel index in the moving image
    max_pixel_size=20.0,      # µm/px; selects pyramid level for alignment
    n_keypoints=10000,
    auto_mask=True,
    thumbnail_max_size=2000,
    qc_out_dir="qc",
    map_rois=False,
    dry_run=False,            # True → QC plots only, skip warping
    only_affine=False,        # True → skip elastix, affine only
    sample_size_factor=3.0,
)

aligner = LocalPalomeroAligner(
    path_from="/path/to/reference.ome.tiff",
    path_to="/path/to/moving.ome.tiff",
    out_path="/path/to/output-registered.ome.tiff",
    task=task,
    temp_zarr_store_dir=None,   # or a path for intermediate zarr cache
)
aligner.run()
```

Workflow steps executed by `aligner.run()`:

1. Read both images and build alignment-resolution pyramids.
2. Coarse affine alignment (ORB feature matching via palom).
3. Non-rigid refinement with itk-elastix (skipped if `only_affine=True`).
4. QC plots saved to `task.qc_out_dir`.
5. Warp moving image and write pyramidal OME-TIFF (skipped if `dry_run=True`).

## See also

Other WSI registration tools, primarily targeting non-same-section alignment
(e.g., serial sections with significant morphological differences):

- **[VALIS](https://github.com/MathOnco/valis)** — fully automated Python
  pipeline supporting 300+ image formats, rigid + non-rigid multi-level
  registration, alignment error estimation, and polygon/cell-coordinate
  warping. Good drop-in choice when you want an end-to-end solution with
  minimal parameter tuning.

- **[DeeperHistReg](https://github.com/MWod/DeeperHistReg)** — deep-learning
  based framework for affine and deformable registration across stains, capable
  of handling very large slides (up to ~220k × 220k px). Docker image
  available for batch use; JSON files drive configuration.

- **[wsireg](https://github.com/NHPatterson/wsireg)** — elastix-based
  registration tool that organizes images and transformations as a graph,
  making it straightforward to chain multi-image or multi-path registrations.
  Can transform associated masks and shape data along registration paths;
  also available as a napari GUI plugin.
