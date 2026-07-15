import csv
import logging
import os
from datetime import datetime, timezone

from config import Config

log = logging.getLogger("results")

RESULTS_CSV = "results.csv"


def log_result(
    config: Config,
    num_params: int,
    initial_loss: float,
    final_loss: float,
    val_loss: float = float("nan"),
    prompt: str = "",
    outcome: str = "",
    device: str = "",
    train_seconds: float = float("nan"),
    prompt_seconds: float = float("nan"),
    notes: str = "",
) -> None:
    utc_now = datetime.now(timezone.utc).replace(microsecond=0)

    fields = {
        "datetime": utc_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": config.run_id,
        **config.log_fields(),
        "num_params": num_params,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "val_loss": round(val_loss, 4),
        "train_seconds": round(train_seconds, 2),
        "prompt_seconds": round(prompt_seconds, 2),
        "device": device,
        "prompt": prompt,
        "outcome": outcome,
        "notes": notes,
    }

    write_header = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if write_header:
            writer.writeheader()
        writer.writerow(fields)

    log.info("Logged run to %s", RESULTS_CSV)
