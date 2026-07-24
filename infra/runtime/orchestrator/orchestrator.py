#!/usr/bin/env python3
"""Supervises a spot training run: relaunch until finished, then stop.

The loop answers one question every `poll_seconds`: is a trainer alive, and is the run
finished? If the answer is "no and no", launch a fresh spot instance -- which will pull
the newest checkpoint from S3 and continue where the dead one left off.

All state lives in S3 and EC2 tags, never in this process. Restart the orchestrator
mid-run and it reattaches to the live trainer instead of launching a duplicate.
"""

from __future__ import annotations

import json
import logging
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

CONFIG_PATH = Path("/etc/llm/orchestrator.json")

# Capacity errors are expected and routine on g6e spot -- they mean "try another AZ", not
# "something is broken". Anything outside this set is a real fault and gets surfaced.
CAPACITY_ERRORS = {
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "InstanceLimitExceeded",
    "Unsupported",
    "UnsupportedOperation",
    "InvalidParameterValue",
    "MaxSpotInstanceCountExceeded",
}

BACKOFF_START = 30
BACKOFF_MAX = 600

log = logging.getLogger("orchestrator")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class Orchestrator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config
        self.run_id: str = config["run_id"]
        self.project: str = config["project"]
        self.bucket: str = config["bucket"]
        self.trainer: dict[str, Any] = config["trainer"]

        self.poll_seconds: int = int(config.get("poll_seconds", 30))
        self.max_run_hours: float = float(config.get("max_run_hours", 48))
        self.max_relaunches: int = int(config.get("max_relaunches", 50))
        self.stall_minutes: float = float(config.get("stall_minutes", 25))

        session = boto3.session.Session(region_name=config["region"])
        self.ec2 = session.client("ec2")
        self.s3 = session.client("s3")

        self.started_at = utcnow()
        self.launches = 0
        self.backoff = BACKOFF_START
        self.az_cursor = 0
        self.running = True

        # AZ -> subnet, ordered. Rotation walks this list.
        self.azs: list[str] = sorted(self.trainer["subnets"].keys())

        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)

    def _stop(self, *_: Any) -> None:
        log.info("shutdown signal received; exiting after this iteration")
        self.running = False

    # --- S3 state ----------------------------------------------------------

    def _key(self, name: str) -> str:
        return f"state/{self.run_id}/{name}"

    def _exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def _put(self, key: str, body: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body.encode())

    def is_done(self) -> bool:
        return self._exists(self._key("DONE"))

    def progress(self) -> dict[str, Any] | None:
        """Latest {step, updated_at} written by the trainer's checkpoint syncer."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self._key("progress.json"))
            return json.loads(obj["Body"].read())
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            log.warning("could not read progress.json: %s", exc)
            return None
        except (ValueError, KeyError) as exc:
            log.warning("progress.json is malformed: %s", exc)
            return None

    def abort(self, reason: str) -> None:
        log.error("ABORTING: %s", reason)
        self._put(
            self._key("ABORTED"),
            json.dumps(
                {
                    "run_id": self.run_id,
                    "reason": reason,
                    "aborted_at": iso(utcnow()),
                    "launches": self.launches,
                },
                indent=2,
            ),
        )

    # --- EC2 ---------------------------------------------------------------

    def find_trainer(self) -> dict[str, Any] | None:
        """The live trainer for this run, if any.

        Tag-based rather than id-based on purpose: after an orchestrator restart there is
        no remembered instance id, but the tags are still on the instance.
        """
        resp = self.ec2.describe_instances(
            Filters=[
                {"Name": "tag:llm:run-id", "Values": [self.run_id]},
                {"Name": "tag:llm:role", "Values": ["trainer"]},
                {"Name": "tag:llm:project", "Values": [self.project]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )
        for reservation in resp["Reservations"]:
            for instance in reservation["Instances"]:
                return instance
        return None

    def terminate(self, instance_id: str, reason: str) -> None:
        log.warning("terminating %s: %s", instance_id, reason)
        try:
            self.ec2.terminate_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            log.error("terminate failed for %s: %s", instance_id, exc)

    def _user_data(self) -> str:
        """Tiny bootstrap: write run.env, start the trainer unit.

        All real logic lives in the AMI so it is version-controlled and reviewable rather
        than smeared across a generated blob.
        """
        env_lines = "\n".join(f"{k}={v}" for k, v in self.cfg["run_env"].items())
        return f"""#!/usr/bin/env bash
set -euo pipefail
install -d -m 0755 /etc/llm
cat > /etc/llm/run.env <<'RUN_ENV_EOF'
{env_lines}
RUN_ENV_EOF
chmod 0600 /etc/llm/run.env
systemctl daemon-reload
systemctl reset-failed llm-trainer.service 2>/dev/null || true
systemctl start llm-trainer.service
"""

    def launch(self) -> str | None:
        """Try each AZ in rotation. Returns the instance id, or None if all are full."""
        t = self.trainer
        tags = [
            {"Key": "Name", "Value": f"{self.project}-trainer"},
            {"Key": "llm:run-id", "Value": self.run_id},
            {"Key": "llm:project", "Value": self.project},
            {"Key": "llm:role", "Value": "trainer"},
            # The orchestrator's IAM policy conditions TerminateInstances on this tag, so
            # a bug in this loop cannot terminate anything it did not create.
            {"Key": "llm:managed-by", "Value": "orchestrator"},
        ]

        # Rotation, not random: retrying the AZ that just reported no capacity is wasted
        # latency, so start from the one after it.
        order = self.azs[self.az_cursor :] + self.azs[: self.az_cursor]

        for az in order:
            subnet = t["subnets"][az]
            log.info("attempting launch in %s (subnet %s)", az, subnet)

            params: dict[str, Any] = {
                "ImageId": t["ami_id"],
                "InstanceType": t["instance_type"],
                "MinCount": 1,
                "MaxCount": 1,
                "SubnetId": subnet,
                "SecurityGroupIds": [t["security_group"]],
                "IamInstanceProfile": {"Name": t["instance_profile"]},
                "UserData": self._user_data(),
                "TagSpecifications": [
                    {"ResourceType": "instance", "Tags": tags},
                    {"ResourceType": "volume", "Tags": tags},
                ],
                "BlockDeviceMappings": [
                    {
                        "DeviceName": "/dev/sda1",
                        "Ebs": {
                            "VolumeSize": int(t["root_volume_gb"]),
                            "VolumeType": "gp3",
                            "DeleteOnTermination": True,
                            "Encrypted": True,
                        },
                    }
                ],
                "MetadataOptions": {
                    "HttpTokens": "required",
                    "HttpEndpoint": "enabled",
                },
                "InstanceMarketOptions": {
                    "MarketType": "spot",
                    "SpotOptions": {
                        # one-time, NOT persistent: a persistent request would auto-replace
                        # the instance behind this loop's back, and only ever in the same
                        # subnet -- defeating AZ rotation entirely.
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                        # No MaxPrice: it defaults to the on-demand price, which is the
                        # recommended setting. A low cap does not save money (you pay the
                        # market rate regardless), it only gets you interrupted more.
                    },
                },
            }
            if t.get("key_name"):
                params["KeyName"] = t["key_name"]

            try:
                resp = self.ec2.run_instances(**params)
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code in CAPACITY_ERRORS:
                    log.warning("no capacity in %s (%s)", az, code)
                    self.az_cursor = (self.azs.index(az) + 1) % len(self.azs)
                    continue
                log.error("launch failed in %s with a non-capacity error: %s", az, exc)
                raise

            instance_id = resp["Instances"][0]["InstanceId"]
            self.launches += 1
            self.az_cursor = (self.azs.index(az) + 1) % len(self.azs)
            self.backoff = BACKOFF_START
            log.info(
                "launched %s in %s (launch %d/%s)",
                instance_id,
                az,
                self.launches,
                self.max_relaunches or "inf",
            )
            self._put(
                self._key(f"launched-{iso(utcnow()).replace(':', '')}.json"),
                json.dumps(
                    {"instance_id": instance_id, "az": az, "launch": self.launches},
                    indent=2,
                ),
            )
            return instance_id

        return None

    # --- Stop conditions ---------------------------------------------------

    def elapsed_hours(self) -> float:
        return (utcnow() - self.started_at).total_seconds() / 3600

    def is_stalled(self, instance: dict[str, Any]) -> bool:
        """True if the trainer is up but has made no progress for stall_minutes.

        This catches the one failure AWS cannot see: an instance that booted, hung (OOM, a
        bad clone, a CUDA fault) and looks perfectly healthy while doing nothing. Without
        it, a wedged trainer burns GPU hours until the wall-clock cap fires.
        """
        launched = instance["LaunchTime"]
        age_minutes = (utcnow() - launched).total_seconds() / 60

        prog = self.progress()
        if prog is None:
            # No progress yet. Boot + venv sync + checkpoint restore + the first checkpoint
            # interval all have to fit in this window, so it is generous on purpose.
            grace = self.stall_minutes * 2
            if age_minutes > grace:
                log.warning(
                    "no progress.json after %.1f min (grace %.1f min)",
                    age_minutes,
                    grace,
                )
                return True
            return False

        try:
            updated = datetime.strptime(
                prog["updated_at"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            log.warning("progress.json has no usable updated_at; ignoring")
            return False

        # Only trust progress written by the instance we are actually looking at -- a stale
        # file from a previous instance would otherwise look like fresh progress.
        if prog.get("instance_id") not in (instance["InstanceId"], None):
            if age_minutes > self.stall_minutes * 2:
                log.warning("progress.json belongs to a previous instance and is stale")
                return True
            return False

        idle = (utcnow() - updated).total_seconds() / 60
        if idle > self.stall_minutes:
            log.warning(
                "no progress for %.1f min (step %s, limit %.1f min)",
                idle,
                prog.get("step"),
                self.stall_minutes,
            )
            return True
        return False

    # --- Main loop ---------------------------------------------------------

    def tick(self) -> bool:
        """One iteration. Returns False when the loop should stop."""
        if self.is_done():
            log.info("DONE marker found -- run %s is complete", self.run_id)
            instance = self.find_trainer()
            if instance:
                self.terminate(instance["InstanceId"], "run complete")
            return False

        if self.elapsed_hours() > self.max_run_hours:
            instance = self.find_trainer()
            if instance:
                self.terminate(instance["InstanceId"], "wall-clock cap")
            self.abort(f"exceeded max_run_hours ({self.max_run_hours})")
            return False

        if self.max_relaunches and self.launches > self.max_relaunches:
            instance = self.find_trainer()
            if instance:
                self.terminate(instance["InstanceId"], "relaunch cap")
            self.abort(f"exceeded max_relaunches ({self.max_relaunches})")
            return False

        instance = self.find_trainer()
        if instance:
            if self.is_stalled(instance):
                # Terminate only; the next tick relaunches. Keeps one decision per tick.
                self.terminate(instance["InstanceId"], "stalled -- no progress")
            else:
                prog = self.progress()
                step = prog.get("step") if prog else "?"
                log.info(
                    "trainer %s %s, step %s, elapsed %.1fh",
                    instance["InstanceId"],
                    instance["State"]["Name"],
                    step,
                    self.elapsed_hours(),
                )
            return True

        log.info("no trainer running -- launching")
        try:
            if self.launch() is None:
                jitter = random.uniform(0.8, 1.2)
                wait = min(self.backoff * jitter, BACKOFF_MAX)
                log.warning(
                    "no capacity in any of %d AZs; backing off %.0fs",
                    len(self.azs),
                    wait,
                )
                time.sleep(wait)
                self.backoff = min(self.backoff * 2, BACKOFF_MAX)
        except ClientError as exc:
            # A non-capacity error (bad AMI, denied permission) will not fix itself by
            # retrying in another AZ. Back off hard rather than hammering the API.
            log.error("launch aborted: %s", exc)
            time.sleep(BACKOFF_MAX)

        return True

    def run(self) -> int:
        log.info(
            "supervising run=%s project=%s azs=%s max_hours=%.1f",
            self.run_id,
            self.project,
            ",".join(self.azs),
            self.max_run_hours,
        )

        while self.running:
            try:
                if not self.tick():
                    break
            except Exception:
                # Never die on a transient AWS error: this process is the only thing that
                # will relaunch the trainer, so staying alive matters more than being right.
                log.exception("unhandled error in tick; continuing")
                time.sleep(self.poll_seconds)
                continue

            time.sleep(self.poll_seconds)

        log.info("orchestrator stopped after %d launches", self.launches)
        return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )

    if not CONFIG_PATH.exists():
        log.error(
            "%s not found -- Terraform user-data should have written it", CONFIG_PATH
        )
        return 1

    config = json.loads(CONFIG_PATH.read_text())

    if not config.get("trainer", {}).get("ami_id"):
        log.error(
            "trainer.ami_id is empty -- build the trainer AMI and re-apply Terraform"
        )
        return 1

    return Orchestrator(config).run()


if __name__ == "__main__":
    sys.exit(main())
