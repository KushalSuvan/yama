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
    parser = argparse.ArgumentParser(description="Run yama models")

    # Required arguments
    parser.add_argument("--model", type=str, required=True, choices=["esum", "mt"],
                        help="which model to run.")
    parser.add_argument("--train-dataset", type=str, default="data/processed/final_train_data.json",
                        help="Path to training dataset JSON (default: config.json path)")

    # Model/training hyperparameters
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for training (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-epochs", type=int, default=8,
                        help="Number of epochs (default: 8)")
    parser.add_argument("--d-model", type=int, default=512,
                        help="Transformer model dimension (default: 512)")
    parser.add_argument("--h", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--N", type=int, default=3,
                        help="Number of encoder layers (default: 3)")
    parser.add_argument("--d-ff", type=int, default=2048,
                        help="Feed-forward network dimension (default: 2048)")
    parser.add_argument("--seq-len", type=int, default=150,
                        help="Maximum sequence length (default: 150)")
    parser.add_argument("--complexity", type=str, choices=["naive", "deep", "attentive"],
                        help="choose the complexity of the extractvie head")

    # Files and paths
    parser.add_argument("--tokenizer-file", type=str, default="tokenizer.json",
                        help="Path to tokenizer file (default: tokenizer.json)")
    parser.add_argument("--model-folder", type=str, default="esum_weights",
                        help="Folder to save/load model weights (default: esum_weights)")
    parser.add_argument("--model-basename", type=str, default="s_model_",
                        help="Base name for model checkpoint files (default: s_model_)")
    parser.add_argument("--preload", type=str, default=None,
                        help="Path to a preloaded model checkpoint (optional)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "esum":
        config = get_config_esum()

        
        config.update({
            "datasource": str(Path(args.train_dataset).resolve()),
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "d_model": args.d_model,
            "h": args.h,
            "N": args.N,
            "d_ff": args.d_ff,
            "seq_len": args.seq_len,
            "tokenizer_file": args.tokenizer_file,
            "model_folder": args.model_folder,
            "model_basename": args.model_basename,
            "preload": args.preload,
            "complexity": args.complexity
        })

        
        # summary_idx = next(iter(train_model_esum(config)))
        # print(summary_idx)

        train_model_esum(config)

    else:
        print(f"Unknown mode: {args.model}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
