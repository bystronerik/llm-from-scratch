import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Model architecture ---
    vocab_size: int = 0  # set once the tokenizer is built
    embedding_dim: int = 64  # size of each token's vector (C)
    context_length: int = 64  # how many tokens the model sees at once (T)
    num_heads: int = 4  # attention heads per transformer block
    num_transformers: int = 6  # number of stacked transformer blocks
    dropout: float = (
        0.1  # fraction of activations zeroed each step; raise if val >> train
    )

    # --- Optimization ---
    batch_size: int = 64  # how many windows per training step (B)
    steps: int = 5000  # number of gradient updates
    learning_rate: float = 1e-3  # peak LR reached at the end of warmup
    warmup_steps: int = 1000  # steps to linearly ramp the LR from 0 to learning_rate
    min_lr: float = 1e-4  # LR floor the cosine decay settles to

    # --- Data / evaluation ---
    num_stories: int = (
        100000  # how many TinyStories entries to concatenate into the corpus
    )
    bpe_merges: int = 512  # how many byte-pair merges the tokenizer learns
    val_fraction: float = 0.1  # portion of the data held out for validation
    eval_interval: int = 500  # how often (in steps) to measure validation loss
    eval_batches: int = 50  # how many val batches to average per measurement

    # --- Paths ---
    # Everything under cache_dir is reproducible run cache: safe to delete, gitignored.
    # tokenizer.json stays versioned in the repo root — delete it to retrain the BPE.
    cache_dir: str = (
        ".cache"  # root for all throwaway run output (datasets, checkpoints)
    )
    run_id: str = ""  # per-run subdir under checkpoints/; set once in build_run()
    tokenizer_path: str = "tokenizer.json"  # cached tokenizer; delete to retrain
    checkpoint_interval: int = 1000  # save a checkpoint every N steps (plus at the end)

    @property
    def dataset_path(self) -> str:
        """Where the downloaded TinyStories corpus is cached."""
        return os.path.join(self.cache_dir, "datasets", "tinystories.txt")

    @property
    def checkpoint_dir(self) -> str:
        """This run's checkpoint directory: .cache/checkpoints/<run_id>."""
        return os.path.join(self.cache_dir, "checkpoints", self.run_id)

    def log_fields(self) -> dict:
        """The subset of settings recorded as columns in results.csv."""
        return {
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }
