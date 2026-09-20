"""Command-Line Interface for the Behavioral Biometrics Zero-Trust Framework."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Authentication via Cursor Dynamics for Zero-Trust Web Architectures."
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "simulate", "ablation", "test"],
        default="benchmark",
        help="Execution mode: 'benchmark' runs all model evaluations, 'simulate' runs live continuous session hijacking, 'ablation' runs feature group analysis, 'test' runs unit tests.",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="user7",
        help="Target user account for continuous hijacking simulation (default: 'user7')",
    )

    args = parser.parse_args()

    if args.mode == "benchmark":
        from experiments.run_benchmark import run_full_benchmark
        run_full_benchmark()

    elif args.mode == "simulate":
        from experiments.run_continuous_simulation import run_simulation
        run_simulation(target_user=args.user)

    elif args.mode == "ablation":
        from experiments.ablation_study import run_ablation_study
        run_ablation_study()

    elif args.mode == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
