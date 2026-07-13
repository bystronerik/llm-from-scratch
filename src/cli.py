import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the language model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from this checkpoint file (e.g. checkpoints/step_1000.pt).",
    )
    return parser.parse_args()
