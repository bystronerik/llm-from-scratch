"""Exercise the orchestrator's decision logic against a fake EC2/S3.

boto3 is stubbed via sys.modules so this runs without the dependency installed.
"""

import json
import sys
import types
from datetime import timedelta
from pathlib import Path
from unittest import mock

# --- stub boto3 before importing the module under test ----------------------


class ClientError(Exception):
    def __init__(self, code, msg="boom"):
        self.response = {"Error": {"Code": code, "Message": msg}}
        super().__init__(f"{code}: {msg}")


boto3_stub = types.ModuleType("boto3")
session_mod = types.ModuleType("boto3.session")


class _Session:
    def __init__(self, **kw):
        pass

    def client(self, name):
        return FAKE[name]


session_mod.Session = _Session
boto3_stub.session = session_mod
sys.modules["boto3"] = boto3_stub
sys.modules["boto3.session"] = session_mod

exc_mod = types.ModuleType("botocore.exceptions")
exc_mod.ClientError = ClientError
botocore = types.ModuleType("botocore")
botocore.exceptions = exc_mod
sys.modules["botocore"] = botocore
sys.modules["botocore.exceptions"] = exc_mod

FAKE = {"ec2": mock.MagicMock(), "s3": mock.MagicMock()}

ORCH = Path(__file__).resolve().parent.parent / "runtime" / "orchestrator"
sys.path.insert(0, str(ORCH))
spec_path = ORCH / "orchestrator.py"
mod = types.ModuleType("orchestrator")
mod.__file__ = str(spec_path)
exec(compile(spec_path.read_text(), str(spec_path), "exec"), mod.__dict__)

Orchestrator = mod.Orchestrator
utcnow = mod.utcnow

# Several tests deliberately inject AWS failures to prove the loop survives them. The
# resulting log.exception tracebacks are expected output, not test failures -- silence
# them so a real problem stands out.
import logging  # noqa: E402

logging.disable(logging.CRITICAL)

# --- fixtures ---------------------------------------------------------------

CONFIG = {
    "region": "us-east-1",
    "project": "llm-training",
    "run_id": "run-001",
    "bucket": "bkt",
    "poll_seconds": 1,
    "max_run_hours": 48,
    "max_relaunches": 50,
    "stall_minutes": 25,
    "trainer": {
        "ami_id": "ami-123",
        "instance_type": "g6e.xlarge",
        "instance_profile": "prof",
        "security_group": "sg-1",
        "key_name": "kp",
        "root_volume_gb": 120,
        "subnets": {
            "us-east-1a": "subnet-a",
            "us-east-1b": "subnet-b",
            "us-east-1d": "subnet-d",
        },
    },
    "run_env": {"RUN_ID": "run-001", "PIPELINE": "true"},
}

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")


def fresh(**overrides):
    # side_effect survives a plain reset_mock(), which leaks a raising stub into the next
    # test. Clear it explicitly.
    FAKE["ec2"].reset_mock(side_effect=True, return_value=True)
    FAKE["s3"].reset_mock(side_effect=True, return_value=True)
    cfg = json.loads(json.dumps(CONFIG))
    cfg.update(overrides)
    o = Orchestrator(cfg)
    return o


def no_trainer():
    FAKE["ec2"].describe_instances.return_value = {"Reservations": []}


def trainer_running(instance_id="i-abc", age_min=5):
    FAKE["ec2"].describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "State": {"Name": "running"},
                        "LaunchTime": utcnow() - timedelta(minutes=age_min),
                    }
                ]
            }
        ]
    }


def s3_state(done=False, progress=None):
    def head(Bucket, Key):
        if Key.endswith("DONE") and done:
            return {}
        raise ClientError("404")

    FAKE["s3"].head_object.side_effect = head

    def get(Bucket, Key):
        if Key.endswith("progress.json") and progress is not None:
            body = mock.MagicMock()
            body.read.return_value = json.dumps(progress).encode()
            return {"Body": body}
        raise ClientError("NoSuchKey")

    FAKE["s3"].get_object.side_effect = get
    FAKE["s3"].put_object.side_effect = None
    FAKE["s3"].put_object.return_value = {}


def launched(instance_id="i-new"):
    FAKE["ec2"].run_instances.return_value = {
        "Instances": [{"InstanceId": instance_id}]
    }


print("\n=== stop conditions ===")

o = fresh()
s3_state(done=True)
trainer_running("i-live")
cont = o.tick()
check("DONE stops the loop", cont is False)
check(
    "DONE terminates the live trainer",
    FAKE["ec2"].terminate_instances.called
    and FAKE["ec2"].terminate_instances.call_args.kwargs["InstanceIds"] == ["i-live"],
)
check("DONE does not launch", not FAKE["ec2"].run_instances.called)

o = fresh()
s3_state()
no_trainer()
o.started_at = utcnow() - timedelta(hours=99)
cont = o.tick()
check("wall-clock cap stops the loop", cont is False)
check(
    "wall-clock cap writes ABORTED",
    any(
        "ABORTED" in c.kwargs.get("Key", "")
        for c in FAKE["s3"].put_object.call_args_list
    ),
)
check("wall-clock cap does not launch", not FAKE["ec2"].run_instances.called)

o = fresh()
s3_state()
no_trainer()
o.launches = 51
cont = o.tick()
check("relaunch cap stops the loop", cont is False)

print("\n=== launching ===")

o = fresh()
s3_state()
no_trainer()
launched()
cont = o.tick()
check("launches when no trainer exists", FAKE["ec2"].run_instances.called)
check("loop continues after launch", cont is True)
kw = FAKE["ec2"].run_instances.call_args.kwargs
check(
    "uses one-time spot, not persistent",
    kw["InstanceMarketOptions"]["SpotOptions"]["SpotInstanceType"] == "one-time",
    str(kw["InstanceMarketOptions"]),
)
check(
    "sets no MaxPrice (defaults to on-demand cap)",
    "MaxPrice" not in kw["InstanceMarketOptions"]["SpotOptions"],
)
check("requires IMDSv2", kw["MetadataOptions"]["HttpTokens"] == "required")
tags = {t["Key"]: t["Value"] for t in kw["TagSpecifications"][0]["Tags"]}
check(
    "tags llm:managed-by=orchestrator (IAM terminate fence)",
    tags.get("llm:managed-by") == "orchestrator",
    str(tags),
)
check("tags llm:run-id", tags.get("llm:run-id") == "run-001")
check("tags llm:role=trainer", tags.get("llm:role") == "trainer")
check("user-data carries PIPELINE=true", "PIPELINE=true" in kw["UserData"])
check("encrypted root volume", kw["BlockDeviceMappings"][0]["Ebs"]["Encrypted"] is True)

o = fresh()
s3_state()
trainer_running("i-healthy")
launched()
o.tick()
check(
    "does NOT launch a duplicate when a trainer is alive",
    not FAKE["ec2"].run_instances.called,
)
check(
    "does not terminate a healthy trainer", not FAKE["ec2"].terminate_instances.called
)

print("\n=== AZ rotation ===")

o = fresh()
s3_state()
no_trainer()
attempts = []


def run_capacity_then_ok(**kw):
    attempts.append(kw["SubnetId"])
    if len(attempts) < 3:
        raise ClientError("InsufficientInstanceCapacity")
    return {"Instances": [{"InstanceId": "i-third"}]}


FAKE["ec2"].run_instances.side_effect = run_capacity_then_ok
o.tick()
check("retries other AZs on capacity error", len(attempts) == 3, f"attempts={attempts}")
check("tried three distinct subnets", len(set(attempts)) == 3, f"attempts={attempts}")

o = fresh()
s3_state()
no_trainer()
FAKE["ec2"].run_instances.side_effect = ClientError("InsufficientInstanceCapacity")
with mock.patch.object(mod.time, "sleep") as slept:
    cont = o.tick()
check("all AZs full -> keeps looping", cont is True)
check("all AZs full -> backs off", slept.called)
check("all AZs full -> launches counter not incremented", o.launches == 0)

o = fresh()
s3_state()
no_trainer()
cursor_before = o.az_cursor
FAKE["ec2"].run_instances.side_effect = [
    ClientError("InsufficientInstanceCapacity"),
    {"Instances": [{"InstanceId": "i-x"}]},
]
o.tick()
check("cursor advances past the failed AZ", o.az_cursor != cursor_before)

o = fresh()
s3_state()
no_trainer()
FAKE["ec2"].run_instances.side_effect = ClientError("UnauthorizedOperation")
with mock.patch.object(mod.time, "sleep") as slept:
    cont = o.tick()
check("non-capacity error does not spin the API", cont is True and slept.called)
check("non-capacity error tried only one AZ", FAKE["ec2"].run_instances.call_count == 1)

print("\n=== stall detection ===")

o = fresh()
s3_state(
    progress={
        "step": 500,
        "instance_id": "i-abc",
        "updated_at": (utcnow() - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
)
trainer_running("i-abc", age_min=60)
o.tick()
check("recent progress -> healthy", not FAKE["ec2"].terminate_instances.called)

o = fresh()
s3_state(
    progress={
        "step": 500,
        "instance_id": "i-abc",
        "updated_at": (utcnow() - timedelta(minutes=40)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
)
trainer_running("i-abc", age_min=90)
o.tick()
check("stale progress -> terminated", FAKE["ec2"].terminate_instances.called)

o = fresh()
s3_state(progress=None)
trainer_running("i-young", age_min=10)
o.tick()
check(
    "young instance with no progress -> grace period",
    not FAKE["ec2"].terminate_instances.called,
)

o = fresh()
s3_state(progress=None)
trainer_running("i-hung", age_min=120)
o.tick()
check(
    "old instance with no progress -> terminated",
    FAKE["ec2"].terminate_instances.called,
)

o = fresh()
s3_state(
    progress={
        "step": 9000,
        "instance_id": "i-OLD",
        "updated_at": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
)
trainer_running("i-CURRENT", age_min=120)
o.tick()
check(
    "progress from a previous instance is not trusted as liveness",
    FAKE["ec2"].terminate_instances.called,
)

o = fresh()
s3_state(progress={"step": 1, "instance_id": "i-abc", "updated_at": "garbage"})
trainer_running("i-abc", age_min=60)
o.tick()
check(
    "malformed updated_at does not kill a trainer",
    not FAKE["ec2"].terminate_instances.called,
)

print("\n=== resilience ===")

o = fresh()
s3_state()
FAKE["ec2"].describe_instances.side_effect = ClientError("RequestLimitExceeded")
with mock.patch.object(mod.time, "sleep"):
    try:
        o.tick()
        raised = False
    except Exception:
        raised = True
check("transient API error propagates to the run() handler", raised)

o = fresh()
FAKE["ec2"].describe_instances.side_effect = ClientError("RequestLimitExceeded")
s3_state()
o.running = True
calls = {"n": 0}
orig_tick = o.tick


def tick_then_stop():
    calls["n"] += 1
    if calls["n"] >= 3:
        o.running = False
    return orig_tick()


o.tick = tick_then_stop
with mock.patch.object(mod.time, "sleep"):
    rc = o.run()
check("run() survives repeated tick failures", rc == 0 and calls["n"] >= 3)

o = fresh()
s3_state()
trainer_running("i-reattach")
o.tick()
f = FAKE["ec2"].describe_instances.call_args.kwargs["Filters"]
names = {x["Name"]: x["Values"] for x in f}
check(
    "finds trainer by tag, not remembered id",
    names.get("tag:llm:run-id") == ["run-001"]
    and names.get("tag:llm:role") == ["trainer"],
)
check(
    "only considers pending/running",
    set(names.get("instance-state-name", [])) == {"pending", "running"},
)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f_ in FAIL:
        print("  FAILED:", f_)

# Guarded: an unconditional sys.exit() at import time aborts any pytest run that collects
# this file. pyproject.toml also scopes pytest to tests/, but belt and braces.
if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
