import logging
from evaluation.benchmark import BenchmarkRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MeshMind.Ablation")


def run_ablation_study(force: bool = False) -> None:
    """Run the 4-condition memory architecture comparison benchmark."""
    logger.info("Starting MeshMind 4-Condition Ablation Study...")
    
    runner = BenchmarkRunner()
    results = runner.run_all_conditions(force=force)
    
    # Output formatting for terminal
    print("\n" + "=" * 90)
    print("                      MESHMIND ABLATION EXPERIMENT SUMMARY")
    print("=" * 90)
    print(
        f"{'Condition':<18} | "
        f"{'Precision':<10} | "
        f"{'Recall':<10} | "
        f"{'Hallucination':<15} | "
        f"{'Personalization':<15} | "
        f"{'Latency (ms)':<12}"
    )
    print("-" * 90)
    
    for cond, metrics in results.items():
        print(
            f"{cond:<18} | "
            f"{metrics.get('precision', 0.0):<10.4f} | "
            f"{metrics.get('recall', 0.0):<10.4f} | "
            f"{metrics.get('hallucination_rate', 0.0):<15.4f} | "
            f"{metrics.get('personalization_score', 1.0):<15.4f} | "
            f"{metrics.get('avg_latency_ms', 0.0):<12.2f}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MeshMind memory ablation study.")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-running the benchmark, ignoring cached results."
    )
    args = parser.parse_args()
    
    run_ablation_study(force=args.force)
