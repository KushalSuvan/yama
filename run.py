import argparse
import sys
import os
from pathlib import Path


# Add src/ to PYTHONPATH
FILE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = str(Path(FILE_DIR) / "src")
sys.path.append(SRC_PATH)

from extractive_summarizer import train_model as train_model_esum
from extractive_summarizer import get_config as get_config_esum


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline modes.")
    parser.add_argument("--model", type=str, required=True, choices=["esum", "mt"],
                        help="Which pipeline mode to run.")
    parser.add_argument("--train-dataset", type=str, required=True,
                        help="Absolute/Relative path to training dataset JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "esum":
        config = get_config_esum()
        config['datasource'] = str(Path(args.train_dataset).resolve())
        train_model_esum(config)
    else:
        print(f"Unknown mode: {args.model}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
