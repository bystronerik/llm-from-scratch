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


def _raw_slug(repo: str, name: str | None, split: str) -> str:
    parts = [repo.replace("/", "__")]
    if name:
        parts.append(name)
    parts.append(split)
    return "_".join(parts)


def fetch_dataset(
    repo: str,
    split: str = "train",
    *,
    name: str | None = None,
    cache_dir: str,
    materialize_raw: bool = True,
):
    from datasets import load_dataset, load_from_disk

    raw_path = os.path.join(cache_dir, "datasets", "raw", _raw_slug(repo, name, split))
    if materialize_raw and os.path.exists(raw_path):
        log.info("Loading cached raw dataset from %s", raw_path)
        return load_from_disk(raw_path)

    token = _ensure_env()
    hf_cache = os.path.join(cache_dir, "hf")
    label = f"{repo}/{name}" if name else repo
    log.info("Downloading %s (split=%s) from HuggingFace ...", label, split)
    ds = load_dataset(repo, name, split=split, cache_dir=hf_cache, token=token)

    if materialize_raw:
        os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
        ds.save_to_disk(raw_path)
        log.info("Saved raw dataset to %s", raw_path)
        ds = load_from_disk(raw_path)
    return ds
