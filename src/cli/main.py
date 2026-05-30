"""Unified command-line interface for manga-dm.

Subcommands:
    select      Sample selection and data download
    stage1      Single-galaxy RC+DM NFW fitting (Stage 1)
    stage2      Population model inference (Stage 2)
    figures     Generate paper figures
    merge       Merge posterior sample files
    sample      Generate robustness sub-samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="manga",
        description="MaNGA Dark Matter analysis pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.toml",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data directory",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Root result directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")

    # --- select ---
    select_parser = subparsers.add_parser("select", help="Select galaxy sample and download data")
    select_parser.add_argument("--download", action="store_true", help="Download data files")
    select_parser.add_argument(
        "--ifu-file", type=str, default=None, help="Output file for plateifu list"
    )

    # --- stage1 ---
    stage1_parser = subparsers.add_parser("stage1", help="Run Stage 1: single-galaxy fitting")
    stage1_parser.add_argument(
        "--ifu", type=str, default="test", help="'test', 'all', or a specific plateifu"
    )
    stage1_parser.add_argument("--nfw", action="store_true", help="Also run NFW DM fitting")
    stage1_parser.add_argument("--n-cores", type=int, default=None, help="Number of parallel workers")

    # --- stage2 ---
    stage2_parser = subparsers.add_parser("stage2", help="Run Stage 2: population model inference")
    stage2_parser.add_argument("--fit", action="store_true", help="Run the population MCMC fit")
    stage2_parser.add_argument(
        "--quality-cut", type=str, default="recommended", help="Quality filter preset"
    )
    stage2_parser.add_argument("--diagnose", action="store_true", help="Run PSIS diagnostics only")

    # --- figures ---
    figures_parser = subparsers.add_parser("figures", help="Generate paper figures")
    figures_parser.add_argument(
        "--ifu", type=str, nargs="+", default=None, help="One or more plateifu identifiers"
    )
    figures_parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    # --- merge ---
    merge_parser = subparsers.add_parser("merge", help="Merge posterior sample files")
    merge_parser.add_argument("--ifu-file", type=str, required=True, help="Path to plateifu list")

    # --- sample ---
    sample_parser = subparsers.add_parser("sample", help="Generate robustness sub-samples")
    sample_parser.add_argument("--n", type=int, required=True, help="Number of sub-samples to generate")

    # Parse
    args = parser.parse_args(argv)

    try:
        from src.config.settings import init_settings

        init_settings(
            config_path=args.config,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if args.subcommand is None:
        parser.print_help()
        return

    # Dispatch (all stubs for now)
    if args.subcommand == "select":
        _run_select(args)
    elif args.subcommand == "stage1":
        _run_stage1(args)
    elif args.subcommand == "stage2":
        _run_stage2(args)
    elif args.subcommand == "figures":
        _run_figures(args)
    elif args.subcommand == "merge":
        _run_merge(args)
    elif args.subcommand == "sample":
        _run_sample(args)


# ── Dispatchers ─────────────────────────────────────────────────────


def _run_select(args) -> None:
    from src.pipeline.selection import select_and_download

    select_and_download(
        ifu_file=args.ifu_file,
        download=args.download,
    )


def _run_stage1(args) -> None:
    from src.pipeline.stage1 import run_stage1

    run_stage1(
        ifu=args.ifu,
        nfw=args.nfw,
        n_cores=args.n_cores,
        result_dir_override=args.result_dir,
    )


def _run_stage2(args) -> None:
    from src.pipeline.stage2 import run_stage2

    run_stage2(
        fit=args.fit,
        quality_cut=args.quality_cut,
        diagnose=args.diagnose,
        result_dir_override=args.result_dir,
    )


def _run_figures(args) -> None:
    if args.ifu is None:
        print("Specify at least one plate-ifu with --ifu.")
        return
    from pathlib import Path

    from src.viz.paper import (
        plot_m200_c_summary_comparison,
        plot_m200_c_summary_panels,
    )
    from src.viz.rc_curves import (
        plot_rc_fit_summary_comparison,
        plot_rc_fit_summary_panels,
    )
    from src.viz.velocity_maps import (
        plot_velocity_field_comparison,
        plot_velocity_field_panels,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else args.result_dir
    )
    if output_dir is None:
        from src.config.settings import settings

        output_dir = settings.result_dir
    else:
        output_dir = Path(output_dir)

    velocity_output = output_dir / "velocity_field_comparison.png"
    posterior_output = output_dir / "m200_c_comparison.png"
    rcfit_output = output_dir / "rc_fit_summary_comparison.png"

    plateifus = list(args.ifu)
    plot_path = plot_velocity_field_comparison(
        plateifus=plateifus,
        output_path=velocity_output,
    )
    posterior_plot_path = plot_m200_c_summary_comparison(
        plateifus=plateifus,
        output_path=posterior_output,
    )
    posterior_panel_paths = plot_m200_c_summary_panels(
        plateifus=plateifus,
        output_path=posterior_output,
    )
    rcfit_plot_path = plot_rc_fit_summary_comparison(
        plateifus=plateifus,
        output_path=rcfit_output,
    )
    rcfit_panel_paths = plot_rc_fit_summary_panels(
        plateifus=plateifus,
        output_path=rcfit_output,
    )
    velocity_panel_paths = plot_velocity_field_panels(
        plateifus=plateifus,
        output_path=velocity_output,
    )

    print(f"Velocity-field figure saved to {plot_path}")
    print(f"M200/c summary figure saved to {posterior_plot_path}")
    print(f"M200/c panel figures saved to {posterior_panel_paths}")
    print(f"RC-fit summary figure saved to {rcfit_plot_path}")
    print(f"RC-fit panel figures saved to {rcfit_panel_paths}")
    print(f"Velocity-field panel figures saved to {velocity_panel_paths}")


def _run_merge(args) -> None:
    from src.pipeline.stage2 import merge_samples

    merge_samples(
        ifu_file=args.ifu_file,
        result_dir_override=args.result_dir,
    )


def _run_sample(args) -> None:
    from src.pipeline.selection import generate_robustness_sample

    generate_robustness_sample(
        n=args.n,
        result_dir_override=args.result_dir,
    )


if __name__ == "__main__":
    main()
