import os
import sys
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.benchmark import BenchmarkRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    runner = BenchmarkRunner()
    runner.run_all_conditions()


if __name__ == "__main__":
    main()
