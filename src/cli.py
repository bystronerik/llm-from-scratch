import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the language model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from a checkpoint file (e.g. .cache/checkpoints/20260713-213500/step_1000.pt).",
    )
    return parser.parse_args()
