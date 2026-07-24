#!/usr/bin/env bash
#
# Bakes the orchestrator AMI. Runs as root on the build instance.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

readonly APP=/opt/llm

log "installing packages"
apt-get update -qq
apt-get install -y -qq \
  python3 \
  python3-boto3 \
  jq \
  curl

python3 -c "import boto3; print('boto3', boto3.__version__)"

log "installing orchestrator"
install -d -m 0755 "$APP/bin" /etc/llm

install -m 0755 /tmp/runtime/orchestrator.py       "$APP/bin/orchestrator.py"
install -m 0644 /tmp/runtime/systemd/*.service     /etc/systemd/system/

systemctl daemon-reload

# Enabled but not started: there is no /etc/llm/orchestrator.json until Terraform's
# user-data writes it on first boot. The unit is configured with a generous
# StartLimitBurst precisely so these early failures do not trip the rate limiter.
systemctl enable llm-orchestrator.service

rm -rf /tmp/runtime

log "smoke test"

python3 -m py_compile "$APP/bin/orchestrator.py" || { echo "FAIL: orchestrator.py does not compile"; exit 1; }
echo "  orchestrator.py compiles"

# Exercise the real failure path: no config present should exit 1 with a clear message,
# not a traceback. This is exactly the state of the machine on first boot.
#
# Output is captured rather than piped into grep: under `pipefail` the pipeline would
# inherit the exit 1 this test is asserting for, and so report failure on success.
if out=$(python3 "$APP/bin/orchestrator.py" 2>&1); then
  echo "FAIL: orchestrator exited 0 with no config present"
  printf '%s\n' "$out"
  exit 1
fi

if printf '%s\n' "$out" | grep -q "not found"; then
  echo "  missing-config path handled cleanly"
else
  echo "FAIL: orchestrator did not handle a missing config gracefully"
  printf '%s\n' "$out"
  exit 1
fi

systemctl is-enabled llm-orchestrator.service >/dev/null || { echo "FAIL: unit not enabled"; exit 1; }
echo "  llm-orchestrator.service enabled"

log "cleaning up"
apt-get clean
rm -rf /var/lib/apt/lists/*
cloud-init clean --logs 2>/dev/null || true
rm -f /etc/ssh/ssh_host_*
rm -f /root/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
rm -rf /tmp/* /var/tmp/*
history -c 2>/dev/null || true

log "orchestrator AMI provisioning complete"
