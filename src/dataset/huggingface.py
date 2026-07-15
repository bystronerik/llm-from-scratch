import logging
import os

log = logging.getLogger("dataset.hf")

_ENV_LOADED = False


def _ensure_env() -> str | None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        from dotenv import load_dotenv

        load_dotenv()
        _ENV_LOADED = True
    return os.environ.get("HF_TOKEN")


def fetch_dataset(
    repo: str,
    name: str | None,
    cache_dir: str,
):
    from datasets import load_dataset, load_from_disk

    raw_path = os.path.join(cache_dir, "datasets", "raw", repo.replace("/", "_"))
    if os.path.exists(raw_path):
        log.info("Loading cached raw dataset from %s", raw_path)
        return load_from_disk(raw_path)

    token = _ensure_env()

    hf_cache = os.path.join(cache_dir, "hf")

    log.info("Downloading %s from HuggingFace ...", repo)

    ds = load_dataset(repo, name, cache_dir=hf_cache, token=token)

    os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)

    ds.save_to_disk(raw_path)

    log.info("Saved raw dataset to %s", raw_path)

    ds = load_from_disk(raw_path)

    return ds
