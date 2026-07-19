import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Tokenizer ---
    tokenizer_path: str = "tokenizer.json"  # cached tokenizer; delete to retrain
    bpe_merges: int = 4096  # how many byte-pair merges the tokenizer learns

    # --- Model architecture ---
    vocab_size: int = 0  # set once the tokenizer is built
    embedding_dim: int = 512  # size of each token's vector (C)
    context_length: int = 128  # how many tokens the model sees at once (T)
    num_heads: int = 8  # attention heads per transformer block
    num_transformers: int = 6  # number of stacked transformer blocks
    dropout: float = (
        0.05  # fraction of activations zeroed each step; raise if val >> train
    )

    # --- Optimization ---
    batch_size: int = 128  # how many windows per training step (B)
    steps: int = 5000  # number of gradient updates
    learning_rate: float = 1e-3  # peak LR reached at the end of warmup
    warmup_steps: int = 1000  # steps to linearly ramp the LR from 0 to learning_rate
    min_lr: float = 1e-4  # LR floor the cosine decay settles to

    # --- Sampling ---
    max_new_tokens: int = 300  # how many tokens to generate for the sample
    temperature: float = 1.0  # <1 sharpens, >1 flattens; 0 = greedy (argmax)
    top_k: int | None = None  # keep only the k most likely tokens, or None
    top_p: float | None = None  # keep the smallest nucleus summing to p, or None
    sink_tokens: int = 4  # earliest tokens the KV cache never evicts
    kv_window: int = 0  # recent tokens kept alongside the sinks; 0 = fill the context

    # --- Data / evaluation ---
    num_articles: int = 100000  # Wikipedia articles streamed into the pretrain corpus (see wikipedia.py)
    val_fraction: float = 0.1  # portion of the data held out for validation
    eval_interval: int = 1000  # how often (in steps) to measure validation loss
    eval_batches: int = 50  # how many val batches to average per measurement

    # --- Caching ---
    cache_dir: str = (
        ".cache"  # root for all throwaway run output (datasets, checkpoints)
    )
    checkpoint_interval: int = 1000  # save a checkpoint every N steps (plus at the end)

    run_id: str = ""  # automatically set once in build_run()

    @property
    def alpaca_path(self) -> str:
        return os.path.join(self.cache_dir, "datasets", "alpaca.txt")

    @property
    def wikipedia_path(self) -> str:
        return os.path.join(self.cache_dir, "datasets", "wikipedia.txt")

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
