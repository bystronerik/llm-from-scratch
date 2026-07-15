import argparse
import logging

from training.command import train_command
from inference.command import run_command

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm", description="Train and run the language model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Pretrain the model (or resume).")
    train.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from a checkpoint file (e.g. .cache/checkpoints/20260713-213500/step_1000.pt).",
    )
    train.set_defaults(func=train_command)

    run = subparsers.add_parser("run", help="Generate text from a trained checkpoint.")
    run.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Checkpoint to load. Defaults to the newest under .cache/checkpoints.",
    )
    run.add_argument(
        "--prompt",
        type=str,
        default="The ",
        help="Text to continue from.",
    )
    run.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Tokens to generate. Defaults to the checkpoint's config.",
    )
    run.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="<1 sharpens, >1 flattens, 0 = greedy. Defaults to the config.",
    )
    run.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only the k most likely tokens. Defaults to the config.",
    )
    run.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold. Defaults to the config.",
    )
    run.set_defaults(func=run_command)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
