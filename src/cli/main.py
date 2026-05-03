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


# ── Stub dispatchers (replaced in later Waves) ──────────────────────────

def _run_select(args):
    print("Not yet implemented: manga select")


def _run_stage1(args):
    print("Not yet implemented: manga stage1")


def _run_stage2(args):
    print("Not yet implemented: manga stage2")


def _run_figures(args):
    print("Not yet implemented: manga figures")


def _run_merge(args):
    print("Not yet implemented: manga merge")


def _run_sample(args):
    print("Not yet implemented: manga sample")


if __name__ == "__main__":
    main()
