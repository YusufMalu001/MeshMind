# --- MONKEY PATCH TO BYPASS PYARROW/DATASETS CRASH ---
import sys
from types import ModuleType
import importlib.machinery

if 'datasets' not in sys.modules:
    class MockDatasets(ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__spec__ = importlib.machinery.ModuleSpec(name, None)

        def __getattr__(self, name):
            if name == '__version__':
                return "3.0.0"
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            class Dummy:
                pass
            Dummy.__name__ = name
            return Dummy

    sys.modules['datasets'] = MockDatasets('datasets')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------------------------------------

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
