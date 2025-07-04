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

from . import omero_handler
from .align.aligner import OmeroRoiAligner
from .models import AlignmentTask, AlignmentResult

log = logging.getLogger(__name__)


def prepare_batch_tasks(args: argparse.Namespace) -> List[AlignmentTask]:
    """Reads CSV and prepares a list of AlignmentTask objects."""
    log.info(f"Reading batch tasks from: {args.batch_csv}")
    tasks: List[AlignmentTask] = []
    required_headers = ["image_id_from", "image_id_to"]
    task_annot = dict(
        filter(
            lambda x: (x[0] not in required_headers)
            & (x[1] in [str, bool, int, float]),
            inspect.get_annotations(AlignmentTask).items(),
        )
    )
    try:
        with open(args.batch_csv, mode="r", encoding="utf-8-sig") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                raise ValueError("CSV file appears to be empty or has no header.")
            if not all(h in reader.fieldnames for h in required_headers):
                missing = set(required_headers) - set(reader.fieldnames)
                raise ValueError(f"CSV missing required headers: {', '.join(missing)}")

            for i, row in enumerate(reader):
                row_num = i + 2
                try:
                    kwargs = {
                        "image_id_from": int(row["image_id_from"]),
                        "image_id_to": int(row["image_id_to"]),
                    }
                    for kk, vv in task_annot.items():
                        kwargs[kk] = vv(row.get(kk) or getattr(args, kk))
                    kwargs["row_num"] = row_num

                    tasks.append(AlignmentTask(**kwargs))
                except (ValueError, TypeError, KeyError) as ve:
                    log.warning(
                        f"Skipping CSV row {row_num} due to invalid value or missing key: {ve}. Row data: {row}"
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
        log.warning("Failures occurred:")
        print("\nFailures occurred:", file=sys.stderr)
        failed_results.sort(
            key=lambda r: r.row_num if r.row_num is not None else float("inf")
        )
        for result in failed_results:
            pair_label = f"from_{result.image_id_from}_to_{result.image_id_to}"
            row_info = f"(CSV Row {result.row_num})" if result.row_num else ""
            log.warning(f"  - Pair {pair_label} {row_info}: {result.message}")
            print(
                f"  - Pair {pair_label} {row_info}: {result.message}", file=sys.stderr
            )
    print(f"\nTotal execution time: {duration:.2f} seconds")


def main():
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s 0.2.0",
    )

    args = parser.parse_args()

    if args.image_id_from is not None and args.image_id_to is None:
        parser.error("--image-id-to is required when --image-id-from is provided.")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )

    conn: Optional[BlitzGateway] = None
    successful_results: List[AlignmentResult] = []
    failed_results: List[AlignmentResult] = []
    start_time = time.time()

    try:
        conn = omero_handler.get_omero_connection()
        if not conn:
            sys.exit(1)

        if args.batch_csv:
            tasks = prepare_batch_tasks(args)
            total_tasks = len(tasks)
            if total_tasks == 0:
                log.info("No valid tasks found in CSV. Exiting.")
            else:
                log.info(
                    f"Starting sequential batch processing for {total_tasks} task(s)."
                )
                for task in tqdm.tqdm(tasks, desc="Processing Batch Sequentially"):
                    if not conn.keepAlive():
                        log.error(
                            "OMERO connection lost during batch processing. Aborting."
                        )
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
                    aligner = OmeroRoiAligner(conn, task)
                    aligner.execute(plot=True)
        else:
            if args.image_id_to is None or args.image_id_from is None:
                parser.error(
                    "Internal error: image_id_to or image_id_from missing for single mode."
                )
            log.info(
                f"Starting Single Pair Mode for: from {args.image_id_from} to {args.image_id_to}"
            )
            task = AlignmentTask(
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
                row_num=None,
            )
            aligner = OmeroRoiAligner(conn, task)
            aligner.execute(plot=True)

    except Exception as e:
        log.error(f"Script execution failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn and conn.isConnected():
            try:
                # conn.close()
                log.info("OMERO connection closed.")
            except Exception as close_e:
                log.warning(f"Error closing OMERO session: {close_e}")

    duration = time.time() - start_time
    report_summary(successful_results, failed_results, duration)

    if failed_results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
