from dataclasses import dataclass

import torch

from config import Config
from model import SimpleLanguageModel
from tokenizer import BPETokenizer
from training import CosineWarmupScheduler


@dataclass
class TrainingState:
    config: Config
    model: SimpleLanguageModel
    optimizer: torch.optim.Optimizer
    scheduler: CosineWarmupScheduler
    tokenizer: BPETokenizer
    start_step: int
