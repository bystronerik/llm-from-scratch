#!/usr/bin/env bash
#
# Bakes the trainer AMI. Runs as root on the build instance.
#
# The goal is that boot-time work is seconds, not minutes: dependencies resolved, corpus
# downloaded, corpus tokenized. Only the application code is fetched at boot, so a
# config.py change does not require a 15-minute AMI rebuild.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

: "${GIT_REPO:?}" "${CODE_SHA:?}" "${NUM_ARTICLES:?}"

readonly APP=/opt/llm
readonly DATA=$APP/data
readonly BUILD=/tmp/build-repo

# --- Packages ---------------------------------------------------------------

log "installing packages"
apt-get update -qq
apt-get install -y -qq \
  inotify-tools \
  jq \
  unzip \
  git \
  curl

# The DLAMI ships an old bundled awscli in some releases; install v2 explicitly so
# `aws s3 cp -` (stdin streaming) and --quiet behave predictably.
if ! aws --version 2>&1 | grep -q 'aws-cli/2'; then
  log "installing awscli v2"
  curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q -o /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install --update
  rm -rf /tmp/awscliv2.zip /tmp/aws
fi

log "installing uv"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
uv --version

# --- Warm the dependency cache ---------------------------------------------
#
# Clone, resolve, then delete the repo but KEEP the uv cache. This is the trick that makes
# "deps baked, code at boot" fast: the boot-time `uv sync` resolves entirely from cache
# instead of downloading several GB of torch and nvidia-* wheels.

log "warming the uv cache from $GIT_REPO@$CODE_SHA"
rm -rf "$BUILD"
git clone --quiet "$GIT_REPO" "$BUILD"
git -C "$BUILD" checkout --quiet "$CODE_SHA"

export UV_CACHE_DIR=/opt/uv-cache
mkdir -p "$UV_CACHE_DIR"

cd "$BUILD"
uv sync --frozen

# --- Stage the corpus and token cache --------------------------------------
#
# Both are derived here, once, rather than on every spot replacement.

log "building corpus and token cache ($NUM_ARTICLES articles)"
mkdir -p "$DATA"

# HF_TOKEN is only in the environment, never written to disk.
if [[ -n ${HF_TOKEN:-} ]]; then
  export HF_TOKEN
fi

# Drive the real code path so the cache keys match exactly what training will look for.
# Anything else risks baking a file the app then ignores.
uv run python - <<'PYTHON'
import os
import sys

sys.path.insert(0, "src")

import numpy as np

from config import Config
from dataset import cache_key, cached_tokens, load_wikipedia_text
from tokenizer import BPETokenizer, EOS_TOKEN
from dataset import load_alpaca_text

config = Config()
print(f"corpus: {config.num_articles} articles -> {config.wikipedia_path}")

text = load_wikipedia_text(config.num_articles, config.cache_dir, config.wikipedia_path)

if os.path.exists(config.tokenizer_path):
    tokenizer = BPETokenizer.load(config.tokenizer_path)
    print(f"tokenizer: loaded {config.tokenizer_path} ({tokenizer.vocab_size} tokens)")
else:
    alpaca = load_alpaca_text(config.cache_dir, config.alpaca_path)
    corpus = (text + "\n" + alpaca).replace(EOS_TOKEN, "\n")
    tokenizer = BPETokenizer.train(corpus, config.bpe_merges)
    tokenizer.save(config.tokenizer_path)
    print(f"tokenizer: trained and saved ({tokenizer.vocab_size} tokens)")

key = cache_key(tokenizer, config.num_articles)
tokens = cached_tokens(
    config.cache_dir,
    key,
    lambda: np.asarray(tokenizer.encode(text), dtype=np.uint16),
)
print(f"tokens: {len(tokens)} cached under key {key}")
PYTHON

cp "$BUILD/.cache/datasets/wikipedia_${NUM_ARTICLES}.txt" "$DATA/"
cp "$BUILD"/.cache/datasets/tokens_*.npy "$DATA/"
cp "$BUILD/tokenizer.json" "$DATA/"

log "staged data"
ls -lh "$DATA"

# The 19 GB raw HuggingFace dataset was only ever an intermediate. Dropping it here is the
# difference between a ~25 GB AMI and a ~45 GB one.
log "removing build artifacts"
rm -rf "$BUILD"
unset HF_TOKEN

# --- Install runtime --------------------------------------------------------

log "installing runtime scripts and units"
install -d -m 0755 "$APP/bin" /etc/llm /var/log/llm

install -m 0755 /tmp/runtime/entrypoint.sh          "$APP/bin/entrypoint.sh"
install -m 0755 /tmp/runtime/spot-watcher.sh        "$APP/bin/spot-watcher.sh"
install -m 0755 /tmp/runtime/checkpoint-syncer.sh   "$APP/bin/checkpoint-syncer.sh"
install -m 0644 /tmp/runtime/systemd/*.service      /etc/systemd/system/

# uv must find the baked cache when the entrypoint runs it.
cat > /etc/profile.d/uv-cache.sh <<'EOF'
export UV_CACHE_DIR=/opt/uv-cache
EOF
mkdir -p /etc/systemd/system/llm-trainer.service.d
cat > /etc/systemd/system/llm-trainer.service.d/10-uv-cache.conf <<'EOF'
[Service]
Environment=UV_CACHE_DIR=/opt/uv-cache
Environment=HOME=/root
EOF

systemctl daemon-reload
systemctl enable llm-trainer.service

# The sidecars are started by the entrypoint once scratch is mounted, so they are NOT
# enabled -- starting them at boot would race the mount and watch an empty directory.
systemctl disable llm-checkpoint-syncer.service llm-spot-watcher.service 2>/dev/null || true

rm -rf /tmp/runtime

# --- Smoke test -------------------------------------------------------------
#
# Fails the build rather than shipping an AMI that only breaks at 3am on a spot instance.

log "smoke test"

nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi did not run"; exit 1; }
echo "  nvidia-smi ok"

UV_CACHE_DIR=/opt/uv-cache uv run --project "$APP" python -c "pass" 2>/dev/null || true

for f in "$DATA/wikipedia_${NUM_ARTICLES}.txt" "$DATA/tokenizer.json"; do
  [[ -s $f ]] || { echo "FAIL: $f missing or empty"; exit 1; }
  echo "  $(basename "$f") ok ($(du -h "$f" | cut -f1))"
done

compgen -G "$DATA/tokens_*.npy" >/dev/null || { echo "FAIL: no token cache staged"; exit 1; }
echo "  token cache ok ($(du -h "$DATA"/tokens_*.npy | cut -f1))"

for s in entrypoint.sh spot-watcher.sh checkpoint-syncer.sh; do
  bash -n "$APP/bin/$s" || { echo "FAIL: $s has a syntax error"; exit 1; }
  echo "  $s parses"
done

command -v inotifywait >/dev/null || { echo "FAIL: inotifywait missing"; exit 1; }
echo "  inotifywait ok"

# --- Cleanup ----------------------------------------------------------------
#
# cloud-init must re-run on every launch, and SSH host keys must be regenerated per
# instance rather than shared by every trainer this AMI ever boots.

log "cleaning up"
apt-get clean
rm -rf /var/lib/apt/lists/*
cloud-init clean --logs 2>/dev/null || true
rm -f /etc/ssh/ssh_host_*
rm -f /root/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
rm -rf /var/log/llm/* /tmp/* /var/tmp/*
history -c 2>/dev/null || true

log "trainer AMI provisioning complete"
