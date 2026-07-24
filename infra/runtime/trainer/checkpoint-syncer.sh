#!/usr/bin/env bash
#
# Uploads checkpoints to S3 as they appear, and prunes old ones.
#
# This is what makes the hybrid storage design safe. It watches for IN_CLOSE_WRITE, which
# fires when torch.save closes the file -- so a 254 MB tensor file is never uploaded
# half-written, and the upload starts the instant the data is complete, independent of
# whatever the training process does next.
#
# That independence is the point: on a spot interruption the checkpoint reaches S3 even if
# the graceful shutdown path overruns its 2-minute budget and gets killed.
set -euo pipefail

readonly RUN_ENV=/etc/llm/run.env
readonly WATCH_DIR=/mnt/scratch/checkpoints

# Retention: newest N always, plus every MILESTONE_EVERY-th step as a permanent marker.
readonly KEEP_RECENT=3
readonly MILESTONE_EVERY=5000

log() { printf '%s [syncer] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# shellcheck disable=SC1090
set -a; source "$RUN_ENV"; set +a
: "${RUN_ID:?}" "${S3_BUCKET:?}"
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"

readonly S3_CHECKPOINTS="s3://$S3_BUCKET/checkpoints/$RUN_ID"
readonly S3_STATE="s3://$S3_BUCKET/state/$RUN_ID"

INSTANCE_ID=$(curl -sf -H "X-aws-ec2-metadata-token: $(curl -sf -X PUT \
  http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')" \
  http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)

mkdir -p "$WATCH_DIR"

step_of() { sed -n 's#.*/step_\([0-9][0-9]*\)\.pt$#\1#p' <<< "$1"; }

upload() {
  local path=$1 name step
  name=$(basename "$path")
  step=$(step_of "$path")
  [[ -z $step ]] && return 0

  log "uploading $name ($(du -h "$path" | cut -f1))"
  if ! aws s3 cp --quiet --only-show-errors "$path" "$S3_CHECKPOINTS/$name"; then
    log "ERROR: upload failed for $name -- will retry on the next checkpoint"
    return 1
  fi

  # progress.json is the orchestrator's liveness signal. Written only after a successful
  # upload, so "progress is advancing" always implies "data is durable".
  printf '{"run_id":"%s","step":%s,"instance_id":"%s","updated_at":"%s"}\n' \
    "$RUN_ID" "$step" "$INSTANCE_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | aws s3 cp --quiet - "$S3_STATE/progress.json" || true

  log "uploaded $name (step $step)"
}

# Bounds S3 storage without waiting on the 30-day lifecycle rule.
prune() {
  local keys step keep=()
  mapfile -t keys < <(aws s3api list-objects-v2 --bucket "$S3_BUCKET" \
    --prefix "checkpoints/$RUN_ID/" --query 'Contents[].Key' --output text 2>/dev/null \
    | tr '\t' '\n' | grep -E 'step_[0-9]+\.pt$' || true)

  [[ ${#keys[@]} -le $KEEP_RECENT ]] && return 0

  # Newest KEEP_RECENT survive regardless.
  mapfile -t keep < <(printf '%s\n' "${keys[@]}" \
    | sed -n 's#.*step_\([0-9][0-9]*\)\.pt$#\1#p' | sort -rn | head -"$KEEP_RECENT")

  local key
  for key in "${keys[@]}"; do
    step=$(step_of "$key")
    [[ -z $step ]] && continue
    # Milestones are kept forever -- cheap insurance for inspecting a run after the fact.
    (( step % MILESTONE_EVERY == 0 )) && continue
    printf '%s\n' "${keep[@]}" | grep -qx "$step" && continue

    log "pruning step $step"
    aws s3 rm --quiet "s3://$S3_BUCKET/$key" || true
  done
}

# Anything already on disk from a previous boot, or written before inotify attached.
log "syncing pre-existing checkpoints in $WATCH_DIR"
shopt -s nullglob
for f in "$WATCH_DIR"/step_*.pt; do upload "$f" || true; done
shopt -u nullglob

log "watching $WATCH_DIR for new checkpoints"

# --format so the loop reads one absolute path per line. close_write (not create) is what
# guarantees the file is complete.
inotifywait -m -q -e close_write --format '%w%f' "$WATCH_DIR" | while read -r path; do
  case $path in
    *.pt)
      upload "$path" || continue
      prune || true
      ;;
  esac
done
