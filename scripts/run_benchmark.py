import logging
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
