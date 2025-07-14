"""Command-line interface for Palomero."""

import argparse
import csv
import inspect
import logging
import os
import sys
import time
from typing import List, Optional

import tqdm
from omero.gateway import BlitzGateway

from . import __version__, omero_handler
from .align.aligner import OmeroRoiAligner
from .models import AlignmentResult, AlignmentTask

log = logging.getLogger(__name__)


def configure_matplotlib_backend():
    """Configures the matplotlib backend."""
    import platform

    import matplotlib

    try:
        # Try to use user-specific config first
        matplotlib.use(matplotlib.get_backend())
        log.info(f"Using matplotlib backend: {matplotlib.get_backend()}")
    except Exception:
        log.warning(
            "User-specific matplotlib backend not available. "
            "Falling back to platform defaults."
        )
        system = platform.system()
        if system == "Linux":
            try:
                matplotlib.use("Agg")
                log.info("Using 'Agg' backend on Linux.")
            except ImportError:
                log.error("Failed to import 'Agg' backend on Linux.")
        elif system == "Darwin":  # macOS
            try:
                matplotlib.use("TkAgg")
                log.info("Using 'TkAgg' backend on macOS.")
            except ImportError:
                log.warning("Failed to import 'TkAgg' backend on macOS. Trying 'Agg'.")
                try:
                    matplotlib.use("Agg")
                    log.info("Using 'Agg' backend on macOS as a fallback.")
                except ImportError:
                    log.error("Failed to import 'Agg' backend on macOS.")
        elif system == "Windows":
            try:
                matplotlib.use("TkAgg")
                log.info("Using 'TkAgg' backend on Windows.")
            except ImportError:
                log.error("Failed to import 'TkAgg' backend on Windows.")


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the CLI."""
    os.environ["COLUMNS"] = "80"
    parser = argparse.ArgumentParser(
        description="Align two OMERO images and transfer ROIs between them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--batch-csv",
        metavar="FILE",
        help="Path to CSV file for batch processing.",
    )
    mode_group.add_argument(
        "--image-id-from",
        type=int,
        metavar="ID",
        help="OMERO ID of the image to transfer ROIs from.",
    )
    parser.add_argument(
        "--image-id-to",
        type=int,
        metavar="ID",
        help="OMERO ID of the image to transfer ROIs to.",
    )
    parser.add_argument(
        "--channel-from",
        type=int,
        default=0,
        metavar="CH",
        help="Channel index for the 'from' image.",
    )
    parser.add_argument(
        "--channel-to",
        type=int,
        default=0,
        metavar="CH",
        help="Channel index for the 'to' image.",
    )
    parser.add_argument(
        "--max-pixel-size",
        type=float,
        default=50.0,
        metavar="MICRONS",
        help="Max pixel size for selecting pyramid level.",
    )
    parser.add_argument(
        "--n-keypoints",
        type=int,
        default=10_000,
        metavar="N",
        help="Number of keypoints for ORB feature detection.",
    )
    parser.add_argument(
        "--auto-mask",
        type=bool,
        default=True,
        metavar="DO_MASK",
        help="Automatically mask out background before image alignment",
    )
    parser.add_argument(
        "--thumbnail-max-size",
        type=int,
        default=2000,
        metavar="MAX_SIZE",
        help="Max thumbnail size when determining image orientations",
    )
    parser.add_argument(
        "--from-mask-roi-id",
        type=int,
        metavar="ID",
        help="ROI ID from the 'from' image to use as a mask.",
    )
    parser.add_argument(
        "--to-mask-roi-id",
        type=int,
        metavar="ID",
        help="ROI ID from the 'to' image to use as a mask.",
    )
    parser.add_argument(
        "--affine-only",
        action="store_true",
        help="Skip non-rigid alignment.",
    )
    parser.add_argument(
        "--map-rois",
        action="store_true",
        help="Map ROIs from 'from' image to 'to' image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run alignment but do not post ROIs.",
    )
    parser.add_argument(
        "--qc-out-dir",
        type=str,
        default="map-roi-qc",
        metavar="DIR",
        help="Output directory for QC plots.",
    )
    parser.add_argument(
        "--close-when-done",
        action="store_true",
        help="Close OMERO connection when all tasks are successful.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def prepare_batch_tasks(args: argparse.Namespace) -> List[AlignmentTask]:
    """Reads CSV and prepares a list of AlignmentTask objects."""
    log.info(f"Reading batch tasks from: {args.batch_csv}")
    tasks: List[AlignmentTask] = []
    required_headers = ["image-id-from", "image-id-to"]
    task_annot = dict(
        filter(
            lambda x: (x[0] not in required_headers)
            and (x[1] in [str, bool, int, float]),
            inspect.get_annotations(AlignmentTask).items(),
        )
    )
    # Manually add Optional[int] fields. The caster will be `int`. If the
    # value is missing from the CSV, it will fall back to the CLI arg,
    # which can be None.
    task_annot["from_mask_roi_id"] = int
    task_annot["to_mask_roi_id"] = int
    try:
        with open(args.batch_csv, mode="r", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                raise ValueError("CSV file appears to be empty or has no header.")
            # replace all - with _ in the header
            reader.fieldnames = [f.replace("-", "_") for f in reader.fieldnames]
            if not all(
                h.replace("-", "_") in reader.fieldnames for h in required_headers
            ):
                missing = set(required_headers) - set(
                    h.replace("_", "-") for h in reader.fieldnames
                )
                raise ValueError(f"CSV missing required headers: {', '.join(missing)}")

            for i, row in enumerate(reader):
                row_num = i + 2
                try:
                    kwargs = {
                        "image_id_from": int(row["image_id_from"]),
                        "image_id_to": int(row["image_id_to"]),
                    }
                    for kk, vv in task_annot.items():
                        val_from_arg = getattr(args, kk)
                        val_from_csv = row.get(kk)

                        if val_from_csv is not None and val_from_csv.strip() != "":
                            caster = vv
                            if caster == bool:
                                caster = lambda x: x.lower() in (
                                    "true",
                                    "1",
                                    "t",
                                    "y",
                                    "yes",
                                )
                            kwargs[kk] = caster(val_from_csv)
                        else:
                            kwargs[kk] = val_from_arg
                    kwargs["row_num"] = row_num
                    tasks.append(AlignmentTask(**kwargs))
                except (ValueError, TypeError, KeyError) as ve:
                    log.warning(
                        f"Skipping CSV row {row_num} due to invalid value: {ve}. Row: {row}"
                    )
                    continue
        log.info(f"Prepared {len(tasks)} tasks from CSV file.")
        return tasks
    except FileNotFoundError:
        log.error(f"Batch CSV file not found: {args.batch_csv}")
        raise
    except Exception as e:
        log.error(f"Failed to read or parse CSV file {args.batch_csv}: {e}")
        raise


def run_task(conn: BlitzGateway, task: AlignmentTask) -> AlignmentResult:
    """Executes a single alignment task and returns the result."""
    try:
        log.info(f"Processing pair: from {task.image_id_from} to {task.image_id_to}")
        aligner = OmeroRoiAligner(conn, task)
        aligner.execute(plot=True)
        return AlignmentResult(
            image_id_from=task.image_id_from,
            image_id_to=task.image_id_to,
            success=True,
            message="Completed successfully.",
            row_num=task.row_num,
        )
    except Exception as e:
        log.error(
            f"Failed to process pair from {task.image_id_from} to {task.image_id_to}: {e}",
            exc_info=True,
        )
        return AlignmentResult(
            image_id_from=task.image_id_from,
            image_id_to=task.image_id_to,
            success=False,
            message=str(e),
            row_num=task.row_num,
        )


def report_summary(
    successful_results: List[AlignmentResult],
    failed_results: List[AlignmentResult],
    duration: float,
):
    """Prints the final summary to the console."""
    total_tasks = len(successful_results) + len(failed_results)
    print("\n--- Processing Summary ---")
    print(f"Total tasks attempted: {total_tasks}")
    print(f"Successful tasks: {len(successful_results)}")
    print(f"Failed tasks: {len(failed_results)}")
    if failed_results:
        print("\nFailures occurred:", file=sys.stderr)
        failed_results.sort(
            key=lambda r: r.row_num if r.row_num is not None else float("inf")
        )
        error_msgs = []
        for result in failed_results:
            pair_label = f"from_{result.image_id_from}_to_{result.image_id_to}"
            row_info = f"(CSV Row {result.row_num})" if result.row_num else ""
            msg = f"Pair {pair_label} {row_info}: {result.message}"
            error_msgs.append(msg)
            print(f"  - {msg}", file=sys.stderr)
        log.warning("Failures occurred:")
        for msg in error_msgs:
            log.warning(msg)
    print(f"\nTotal execution time: {duration:.2f} seconds")


def main():
    """Main entry point for the CLI."""
    configure_matplotlib_backend()
    parser = create_parser()
    args = parser.parse_args()

    if args.image_id_from is not None and args.image_id_to is None:
        parser.error("--image-id-to is required when --image-id-from is provided.")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    successful_results: List[AlignmentResult] = []
    failed_results: List[AlignmentResult] = []
    start_time = time.time()

    conn: Optional[BlitzGateway] = None
    try:
        conn = omero_handler.get_omero_connection()
        if not conn:
            sys.exit(1)

        tasks: List[AlignmentTask] = []
        if args.batch_csv:
            tasks.extend(prepare_batch_tasks(args))
        else:
            tasks.append(
                AlignmentTask(
                    image_id_from=args.image_id_from,
                    image_id_to=args.image_id_to,
                    channel_from=args.channel_from,
                    channel_to=args.channel_to,
                    max_pixel_size=args.max_pixel_size,
                    n_keypoints=args.n_keypoints,
                    auto_mask=args.auto_mask,
                    thumbnail_max_size=args.thumbnail_max_size,
                    qc_out_dir=args.qc_out_dir,
                    map_rois=args.map_rois,
                    dry_run=args.dry_run,
                    affine_only=args.affine_only,
                    from_mask_roi_id=args.from_mask_roi_id,
                    to_mask_roi_id=args.to_mask_roi_id,
                    row_num=None,
                )
            )

        if not tasks:
            log.info("No tasks to process. Exiting.")
        else:
            log.info(f"Starting processing for {len(tasks)} task(s).")
            for task in tqdm.tqdm(tasks, desc="Processing Tasks"):
                if not conn.keepAlive():
                    log.error("OMERO connection lost. Aborting.")
                    failed_results.append(
                        AlignmentResult(
                            image_id_from=task.image_id_from,
                            image_id_to=task.image_id_to,
                            success=False,
                            message="OMERO connection lost.",
                            row_num=task.row_num,
                        )
                    )
                    break
                result = run_task(conn, task)
                if result.success:
                    successful_results.append(result)
                else:
                    failed_results.append(result)

    except Exception as e:
        log.critical(f"A critical error occurred: {e}", exc_info=True)
        print(f"\nError: A critical error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn and conn.isConnected():
            all_successful = not failed_results
            if args.close_when_done and all_successful:
                try:
                    conn.close()
                    log.info("OMERO connection closed.")
                except Exception as close_e:
                    log.warning(f"Error closing OMERO session: {close_e}")
            else:
                log.info("Keeping OMERO connection open.")

    duration = time.time() - start_time
    report_summary(successful_results, failed_results, duration)

    if failed_results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
