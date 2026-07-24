#!/usr/bin/env bash
#
# Trainer boot sequence. Runs as llm-trainer.service.
#
#   config -> scratch -> code -> deps -> data -> restore -> sidecars -> train -> report
#
# Every step is idempotent: this script runs again from scratch on every spot replacement,
# and must reach the same state whether it is the first launch or the twentieth.
set -euo pipefail

readonly RUN_ENV=/etc/llm/run.env
readonly DATA_DIR=/opt/llm/data
readonly REPO_DIR=/opt/llm/repo
readonly SCRATCH=/mnt/scratch
readonly LOG_DIR=/var/log/llm

log() { printf '%s [entrypoint] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

# --- 1. Config --------------------------------------------------------------

[[ -f $RUN_ENV ]] || die "$RUN_ENV missing -- the orchestrator should have written it via user-data."
# shellcheck disable=SC1090
set -a; source "$RUN_ENV"; set +a

: "${RUN_ID:?}" "${S3_BUCKET:?}" "${CODE_SHA:?}" "${GIT_REPO:?}"
: "${DEPLOY_KEY_PARAM:?}" "${AWS_REGION:?}"
export AWS_DEFAULT_REGION="$AWS_REGION"

readonly S3_CHECKPOINTS="s3://$S3_BUCKET/checkpoints/$RUN_ID"
readonly S3_FINAL="s3://$S3_BUCKET/final/$RUN_ID"
readonly S3_STATE="s3://$S3_BUCKET/state/$RUN_ID"
readonly S3_LOGS="s3://$S3_BUCKET/logs/$RUN_ID"

# IMDSv2: the token is mandatory, the metadata service rejects unauthenticated reads.
imds() {
  local token
  token=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300") || return 1
  curl -sf -H "X-aws-ec2-metadata-token: $token" \
    "http://169.254.169.254/latest/meta-data/$1"
}

INSTANCE_ID=$(imds instance-id || echo "unknown")
export INSTANCE_ID
log "run=$RUN_ID instance=$INSTANCE_ID sha=$CODE_SHA"

mkdir -p "$LOG_DIR"

# --- 2. Scratch disk --------------------------------------------------------
#
# g6e ships a local NVMe instance store: free, faster than EBS, and ephemeral -- which is
# exactly right, because S3 is the durable store. Nothing here needs to survive.

mount_scratch() {
  if mountpoint -q "$SCRATCH"; then
    log "scratch already mounted"
    return 0
  fi

  mkdir -p "$SCRATCH"

  # Instance store volumes report this exact model string on Nitro instances. Matching on
  # the model rather than a device path avoids clobbering the EBS root volume, whose
  # device name is not stable across instance types.
  local dev
  dev=$(lsblk -d -n -o NAME,MODEL 2>/dev/null \
    | awk '/Amazon EC2 NVMe Instance Storage/ {print "/dev/"$1; exit}')

  if [[ -z $dev ]]; then
    log "WARNING: no instance-store NVMe found; falling back to the root volume."
    log "         Checkpoint writes will be slower and consume root disk."
    return 0
  fi

  log "formatting instance store $dev"
  mkfs.ext4 -F -m 0 -E nodiscard "$dev" >/dev/null 2>&1 || die "mkfs failed on $dev"
  mount -o noatime,nodiratime "$dev" "$SCRATCH" || die "mount failed for $dev"
  log "scratch mounted: $(df -h "$SCRATCH" | tail -1)"
}

mount_scratch
mkdir -p "$SCRATCH/checkpoints"

# --- 3. Code ----------------------------------------------------------------
#
# The deploy key lives only in tmpfs and only for the duration of the clone.

readonly KEY_FILE=/dev/shm/deploy_key
cleanup_key() { shred -u "$KEY_FILE" 2>/dev/null || rm -f "$KEY_FILE"; }
trap cleanup_key EXIT

fetch_code() {
  log "fetching deploy key from SSM: $DEPLOY_KEY_PARAM"
  local key
  key=$(aws ssm get-parameter --name "$DEPLOY_KEY_PARAM" --with-decryption \
    --query 'Parameter.Value' --output text) || die "could not read $DEPLOY_KEY_PARAM"

  # Fail loudly here rather than hanging on an interactive SSH password prompt later.
  [[ $key == PLACEHOLDER-* ]] && die "deploy key is still the Terraform placeholder. Run: make deploy-key KEY=<path>"

  install -m 0600 /dev/null "$KEY_FILE"
  printf '%s\n' "$key" > "$KEY_FILE"

  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts 2>/dev/null
  sort -u /root/.ssh/known_hosts -o /root/.ssh/known_hosts

  export GIT_SSH_COMMAND="ssh -i $KEY_FILE -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes"

  rm -rf "$REPO_DIR"
  log "cloning $GIT_REPO"
  git clone --quiet "$GIT_REPO" "$REPO_DIR" || die "clone failed"

  git -C "$REPO_DIR" checkout --quiet "$CODE_SHA" || die "checkout $CODE_SHA failed"
  log "checked out $(git -C "$REPO_DIR" rev-parse --short HEAD)"
}

fetch_code
cleanup_key
trap - EXIT

# --- 4. Dependencies --------------------------------------------------------
#
# Seconds, not minutes: the uv cache is baked into the AMI, so this resolves offline
# rather than pulling several GB of torch + nvidia wheels.

cd "$REPO_DIR"
log "syncing dependencies"
uv sync --frozen 2>&1 | tail -3 || die "uv sync failed"

# --- 5. Data ----------------------------------------------------------------
#
# Symlinks, not copies -- the corpus and token cache are ~900 MB and read-only.

link_data() {
  mkdir -p "$REPO_DIR/.cache/datasets"

  # Globbed, not constructed from a NUM_ARTICLES variable: the filename embeds the article
  # count that was actually baked, and a mismatch between that and run.env would silently
  # trigger a full re-download from HuggingFace on a box with no HF_TOKEN.
  local corpus
  corpus=$(find "$DATA_DIR" -maxdepth 1 -name 'wikipedia_*.txt' -print -quit)
  if [[ -n $corpus ]]; then
    ln -sfn "$corpus" "$REPO_DIR/.cache/datasets/$(basename "$corpus")"
    log "corpus linked: $(basename "$corpus")"
  else
    log "WARNING: no baked corpus found -- the app will try to download it."
  fi

  [[ -f $DATA_DIR/tokenizer.json ]] && ln -sfn "$DATA_DIR/tokenizer.json" "$REPO_DIR/tokenizer.json"

  # The token cache is what makes relaunches cheap (see PLAN.md 2.1). Its filename embeds
  # a hash of the tokenizer state, so a tokenizer change silently invalidates it -- worth
  # shouting about, because the only symptom is a slow start.
  local tokens
  tokens=$(find "$DATA_DIR" -maxdepth 1 -name 'tokens_*.npy' -print -quit)
  if [[ -n $tokens ]]; then
    ln -sfn "$tokens" "$REPO_DIR/.cache/datasets/$(basename "$tokens")"
    log "token cache linked: $(basename "$tokens")"
  else
    log "WARNING: no baked token cache found. The corpus will be re-tokenized (~60-90s)."
  fi

  # Checkpoints go to scratch, not the root volume.
  rm -rf "$REPO_DIR/.cache/checkpoints"
  ln -sfn "$SCRATCH/checkpoints" "$REPO_DIR/.cache/checkpoints"
}

link_data

# The authoritative step target is the pinned code's own config, not a value passed in
# through user-data -- resume() rebuilds Config() from disk, so anything else could drift.
MAX_STEPS=$(uv run python -c "import sys; sys.path.insert(0,'src'); from config import Config; print(Config().steps)")
readonly MAX_STEPS
log "target steps: $MAX_STEPS"

# --- 6. Restore -------------------------------------------------------------

# Step numbers present in S3, highest first.
steps_in_s3_desc() {
  aws s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "checkpoints/$RUN_ID/" \
    --query 'Contents[].Key' --output text 2>/dev/null \
    | tr '\t' '\n' \
    | sed -n 's#.*/step_\([0-9][0-9]*\)\.pt$#\1#p' \
    | sort -rn
}

# Verifies the file actually deserializes. A truncated upload otherwise fails 40 minutes
# into training instead of here.
verify_checkpoint() {
  uv run python -c "
import sys
sys.path.insert(0, 'src')
from training.checkpoint import load_checkpoint
print(load_checkpoint(sys.argv[1])['step'])
" "$1" 2>/dev/null
}

RESUME_FROM=""
restore() {
  local steps step local_path
  # Selected by step NUMBER, never by LastModified -- mtime ordering lies after a
  # re-upload, and picking an older checkpoint silently discards training progress.
  mapfile -t steps < <(steps_in_s3_desc)

  if [[ ${#steps[@]} -eq 0 ]]; then
    log "no checkpoint in S3 -- starting a fresh run"
    return 0
  fi

  # Walk downwards so a corrupt newest checkpoint costs one interval, not the whole run.
  for step in "${steps[@]}"; do
    local_path="$SCRATCH/checkpoints/step_${step}.pt"
    log "downloading checkpoint step_${step}.pt"
    if ! aws s3 cp --quiet "$S3_CHECKPOINTS/step_${step}.pt" "$local_path"; then
      log "download failed for step $step; trying the previous one"
      continue
    fi
    if verify_checkpoint "$local_path" >/dev/null; then
      RESUME_FROM="$local_path"
      log "resuming from step $step"
      return 0
    fi
    log "checkpoint step $step failed to load; trying the previous one"
    rm -f "$local_path"
  done

  die "every checkpoint in S3 failed to load -- refusing to silently restart from zero"
}

restore

# --- 7. Sidecars ------------------------------------------------------------

log "starting checkpoint syncer and spot watcher"
systemctl start llm-checkpoint-syncer.service
systemctl start llm-spot-watcher.service

# --- 8. Train ---------------------------------------------------------------

log "starting training (PIPELINE=${PIPELINE:-false})"
TRAIN_RC=0
if [[ -n $RESUME_FROM ]]; then
  ./llm train --checkpoint "$RESUME_FROM" || TRAIN_RC=$?
else
  ./llm train || TRAIN_RC=$?
fi
log "training process exited rc=$TRAIN_RC"

# --- 9. Report --------------------------------------------------------------
#
# Exit code 0 does NOT mean the run finished: a spot interruption sends SIGINT, the app
# catches KeyboardInterrupt, saves a checkpoint and exits cleanly. Completion is decided
# by step count alone.

highest_step_local() {
  find "$SCRATCH/checkpoints" -name 'step_*.pt' 2>/dev/null \
    | sed -n 's#.*/step_\([0-9][0-9]*\)\.pt$#\1#p' | sort -n | tail -1
}

FINAL_STEP=$(highest_step_local)
FINAL_STEP=${FINAL_STEP:-0}
log "highest local checkpoint: step $FINAL_STEP of $MAX_STEPS"

# Give the syncer a moment to drain its inotify queue before we judge what made it up.
sleep 10

if [[ $TRAIN_RC -eq 0 && $FINAL_STEP -ge $MAX_STEPS ]]; then
  log "run complete -- publishing final artifacts"

  aws s3 cp --quiet "$SCRATCH/checkpoints/step_${FINAL_STEP}.pt" "$S3_FINAL/step_${FINAL_STEP}.pt"
  [[ -f $REPO_DIR/results.csv ]] && aws s3 cp --quiet "$REPO_DIR/results.csv" "$S3_FINAL/results.csv"
  [[ -f $REPO_DIR/tokenizer.json ]] && aws s3 cp --quiet "$REPO_DIR/tokenizer.json" "$S3_FINAL/tokenizer.json"

  printf '{"run_id":"%s","step":%s,"instance_id":"%s","completed_at":"%s","commit":"%s"}\n' \
    "$RUN_ID" "$FINAL_STEP" "$INSTANCE_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(git -C "$REPO_DIR" rev-parse HEAD)" \
    | aws s3 cp --quiet - "$S3_FINAL/manifest.json"

  # Written LAST: it is the orchestrator's signal to stop relaunching, so everything it
  # advertises must already be in place.
  printf 'run %s completed at step %s on %s\n' "$RUN_ID" "$FINAL_STEP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | aws s3 cp --quiet - "$S3_STATE/DONE"

  log "DONE marker written"
else
  log "run not finished (rc=$TRAIN_RC, step $FINAL_STEP/$MAX_STEPS) -- the orchestrator will relaunch"
fi

# Archive this boot's log regardless of outcome.
if [[ -f $LOG_DIR/trainer.log ]]; then
  aws s3 cp --quiet "$LOG_DIR/trainer.log" \
    "$S3_LOGS/${INSTANCE_ID}-$(date -u +%Y%m%dT%H%M%SZ).log" || true
fi

log "entrypoint finished; shutting down"
shutdown -h +1 "training wrapper finished"
