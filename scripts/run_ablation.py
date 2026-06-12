import argparse
from experiments.ablation import run_ablation_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MeshMind memory ablation study.")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-running the benchmark, ignoring cached results."
    )
    args = parser.parse_args()
    
    run_ablation_study(force=args.force)


if __name__ == "__main__":
    main()
