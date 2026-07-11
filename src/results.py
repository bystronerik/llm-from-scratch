import csv
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("results")

RESULTS_CSV = "results.csv"


def log_result(
    hyperparams: dict,
    num_params: int,
    initial_loss: float,
    final_loss: float,
    notes: str = "",
) -> None:
    """Append this run's settings and result to results.csv, creating it if needed."""
    utc_now = datetime.now(timezone.utc).replace(microsecond=0)
    fields = {
        "datetime": utc_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **hyperparams,
        "num_params": num_params,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "notes": notes,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        if write_header:
            writer.writeheader()
        writer.writerow(fields)
    log.info("Appended run to %s", RESULTS_CSV)
