#!/usr/bin/env bash
#
# Converts an EC2 spot interruption notice into a graceful checkpoint save.
#
# AWS publishes the notice to IMDS roughly 2 minutes before reclaiming the instance. That
# is the entire warning; nothing else fires. src/training/train.py already catches
# KeyboardInterrupt and does an evaluate-then-checkpoint, so all this needs to do is pull
# the trigger with SIGINT.
#
# Correctness does NOT depend on that graceful path finishing: the checkpoint syncer
# uploads on IN_CLOSE_WRITE, so the file is safe in S3 the moment torch.save returns,
# whatever the process does afterwards.
set -euo pipefail

readonly RUN_ENV=/etc/llm/run.env
readonly POLL_SECONDS=5
readonly IMDS=http://169.254.169.254

log() { printf '%s [spot-watcher] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# shellcheck disable=SC1090
[[ -f $RUN_ENV ]] && { set -a; source "$RUN_ENV"; set +a; }
export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"

imds_token() {
  curl -sf -X PUT "$IMDS/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null
}

INSTANCE_ID=$(curl -sf -H "X-aws-ec2-metadata-token: $(imds_token)" \
  "$IMDS/latest/meta-data/instance-id" 2>/dev/null || echo unknown)

log "watching for interruption notices (every ${POLL_SECONDS}s)"

while true; do
  token=$(imds_token) || { sleep "$POLL_SECONDS"; continue; }

  # 200 = reclaim scheduled. 404 = normal, nothing pending. Anything else = transient.
  notice=$(curl -sf -H "X-aws-ec2-metadata-token: $token" \
    "$IMDS/latest/meta-data/spot/instance-action" 2>/dev/null || true)

  if [[ -n $notice ]]; then
    log "INTERRUPTION NOTICE: $notice"

    # Breadcrumb first -- if the instance dies faster than advertised, this is the only
    # record that it was a reclaim rather than a crash.
    if [[ -n ${S3_BUCKET:-} && -n ${RUN_ID:-} ]]; then
      printf '{"instance_id":"%s","noticed_at":"%s","notice":%s}\n' \
        "$INSTANCE_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$notice" \
        | aws s3 cp --quiet - \
          "s3://$S3_BUCKET/state/$RUN_ID/interrupted-$(date -u +%Y%m%dT%H%M%SZ).json" || true
    fi

    # Signal the training process itself, not the wrapper: SIGINT to the wrapper's process
    # group would kill the syncer mid-upload.
    pid=$(pgrep -f 'src/main.py train' | head -1 || true)
    if [[ -n $pid ]]; then
      log "sending SIGINT to training process $pid"
      kill -INT "$pid" || log "SIGINT failed; process may have already exited"
    else
      log "no training process found -- nothing to interrupt"
    fi

    # The instance is going away regardless. Stop polling.
    log "watcher exiting; the syncer will drain remaining checkpoints"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
