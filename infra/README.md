# AWS training infrastructure

Trains the model on an EC2 **g6e.xlarge spot** instance, supervised by a **t4g.small
orchestrator** that relaunches the trainer after every spot interruption until the run
reaches `config.steps`.

Design rationale and trade-offs live in [PLAN.md](PLAN.md). This file is the runbook.

```
you ── terraform ──> VPC, S3, IAM, orchestrator
    └─ packer ─────> trainer AMI (x86_64), orchestrator AMI (arm64)

orchestrator (t4g.small, always up)
    │  every 30s: is a trainer alive? is the run finished?
    └─ RunInstances (spot, one-time, rotating AZ)
           │
      trainer (g6e.xlarge spot, ephemeral)
           ├─ clone pinned SHA, restore newest checkpoint from S3
           ├─ train, writing checkpoints to instance-store NVMe
           ├─ syncer uploads each checkpoint to S3 on close_write
           └─ spot notice -> SIGINT -> graceful save
```

## First-time setup

Prerequisites: `terraform >= 1.11`, `packer`, `awscli v2`, `jq`, and AWS credentials.

The Terraform is split into independent **state layers** under `terraform/layers/`
(`10-network` … `60-compute`), each composed from a reusable module in `terraform/modules/`.
`make` applies them in dependency order, so day-to-day you still just run `make apply`.
Configuration is split the same way: shared values live in one `common.tfvars`, and each
layer reads its own `terraform.tfvars`.

```bash
cd infra
cp terraform/common.tfvars.example                     terraform/common.tfvars
cp terraform/layers/10-network/terraform.tfvars.example      terraform/layers/10-network/terraform.tfvars
cp terraform/layers/40-observability/terraform.tfvars.example terraform/layers/40-observability/terraform.tfvars
cp terraform/layers/60-compute/terraform.tfvars.example      terraform/layers/60-compute/terraform.tfvars
```

Set the four placeholders across those files:

| Value | File |
|---|---|
| `aws_profile` | `terraform/common.tfvars` |
| `ssh_allowed_cidr` | `terraform/layers/10-network/terraform.tfvars` |
| `alert_email`, `monthly_budget_usd` | `terraform/layers/40-observability/terraform.tfvars` |

The defaults are valid but inert (RFC 5737 / RFC 2606 reserved), so the stack builds without
them; it just won't let you in or email you.

```bash
make bootstrap                                   # state bucket (once per account)
make init                                        # wire the S3 backend of every layer
make keypair                                     # EC2 key pair -> ~/.ssh/llm-training.pem
make apply                                       # all layers: VPC, S3, IAM, SSM  (no compute yet)
make deploy-key KEY=~/.ssh/llm_deploy_key        # GitHub deploy key -> SSM
make ami                                         # both AMIs (~20 min)
make apply                                       # orchestrator starts the run
```

Set `ssh_key_name = "llm-training"` in `terraform/layers/60-compute/terraform.tfvars` after
`make keypair`. To (re)apply just one layer, use `make apply-one LAYER=60-compute`.

### The deploy key

```bash
ssh-keygen -t ed25519 -f ~/.ssh/llm_deploy_key -N '' -C 'llm-training'
```

Add `~/.ssh/llm_deploy_key.pub` to the GitHub repo under **Settings → Deploy keys**, with
write access **off**. Then `make deploy-key KEY=~/.ssh/llm_deploy_key`.

This goes through the AWS CLI rather than Terraform on purpose: Terraform state is stored
in plaintext, so a private key passed as a variable would be readable by anyone who can
read the state bucket.

### HuggingFace token

Only needed at **AMI build time**, to download the Wikipedia dataset. `make ami-trainer`
picks it up from `$HF_TOKEN` or from the repo's `.env`. It is passed to the build as an
environment variable and never written into the image — and the 19 GB raw dataset is
deleted after the 470 MB corpus is derived from it.

The trainer itself needs no HuggingFace access: the corpus, tokenizer and token cache are
all baked in.

## Running

```bash
make status          # instances, run state, current step
make logs            # follow the orchestrator
make trainer-logs    # tail the live trainer
make checkpoints     # what is in S3
make fetch-final     # download the finished model
```

To start a **fresh run**, bump `run_id` in `terraform/layers/60-compute/terraform.tfvars`
and `make apply-one LAYER=60-compute`. To **resume**, leave `run_id` alone — the trainer
picks up the newest checkpoint under that id automatically.

Pin `code_sha` to a full commit for any real run. `resume()` in
[`src/training/setup.py`](../src/training/setup.py) rebuilds a fresh `Config()` rather than
restoring `steps` from the checkpoint, so the LR schedule comes from whatever `config.py`
is on disk — a moving ref could change it underneath a resume.

## Testing before you spend money

```bash
make test        # 37 orchestrator logic tests + syntax checks, no AWS calls
make validate    # terraform + packer
```

The orchestrator tests stub EC2 and S3 entirely, so they cover the decisions that matter —
duplicate-launch prevention, AZ rotation, stall detection, the terminate fence — without
launching anything.

To rehearse a spot interruption for real:

```bash
aws ec2 send-spot-instance-interruptions --instance-id i-xxxx --instance-interruption-time <ts>
```

Then watch a checkpoint land in S3 and a replacement instance resume from it.

## How interruption handling works

`src/training/train.py` already catches `KeyboardInterrupt` and does an
evaluate-then-checkpoint. The spot watcher polls IMDS every 5 s and sends `SIGINT` when AWS
publishes a reclaim notice, which is the entire 2-minute warning you get.

Correctness does **not** depend on that graceful path finishing. The checkpoint syncer
watches for `IN_CLOSE_WRITE`, so a checkpoint starts uploading the moment `torch.save`
closes the file — independent of whatever the process does next. If the shutdown overruns
its budget and gets killed, the checkpoint is already in S3.

`PIPELINE=true` (set in the trainer's `run.env`) skips the post-training text sample, so
the interruption budget is spent saving rather than generating.

## Costs

| | |
|---|---|
| g6e.xlarge spot | ~$0.55–0.90/hr (varies by AZ and time) |
| t4g.small orchestrator | ~$0.017/hr — **the only thing billed while idle** |
| Root EBS 120 GB gp3 | ~$10/month, only while the trainer runs |
| Instance-store NVMe | free |
| S3 | cents |

The orchestrator at ~$12/month is what to watch — it is the only resource that costs money
when nothing is happening. `make destroy` between runs, or leave it and rely on the budget
alarm.

Verify current spot pricing:

```bash
aws ec2 describe-spot-price-history --instance-types g6e.xlarge \
  --product-descriptions Linux/UNIX --max-items 10 \
  --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output table
```

## Layout

Each **module** holds the resources for one concern; each **layer** is a thin root that
composes one module, owns its own Terraform state, and (for `iam`/`compute`) reads upstream
layers via `terraform_remote_state`. Applying in the `10 → 60` order keeps blast radius
small — a budget edit re-plans only `40-observability`, never the VPC.

```
infra/
├── Makefile                     every operation below (walks the layers)
├── terraform/
│   ├── common.tfvars(.example)  shared inputs (project, region, profile, tags)
│   ├── modules/                 reusable, single-concern building blocks
│   │   ├── network/             VPC, dynamic AZ discovery, SGs, S3 endpoint
│   │   ├── storage/             artifact bucket + lifecycle rules
│   │   ├── ssm/                 deploy key parameter (value set out-of-band)
│   │   ├── observability/       monthly budget alarm + CloudWatch log group
│   │   ├── iam/                 scoped trainer and orchestrator roles
│   │   └── compute/             t4g.small + EIP + user-data template
│   └── layers/                  one state each, applied in order
│       ├── 00-bootstrap/        state bucket (local state, run once)
│       ├── 10-network/          -> module.network
│       ├── 20-storage/          -> module.storage
│       ├── 30-ssm/              -> module.ssm
│       ├── 40-observability/    -> module.observability
│       ├── 50-iam/              -> module.iam        (reads 10/20/30/40)
│       └── 60-compute/          -> module.compute    (reads 10/20/30/40/50)
├── packer/                      trainer (x86_64) and orchestrator (arm64) AMIs
├── runtime/
│   ├── trainer/                 entrypoint, spot watcher, checkpoint syncer, units
│   └── orchestrator/            supervision loop + unit
└── tests/                       orchestrator logic tests (no AWS required)
```

## Troubleshooting

**Orchestrator not launching anything.** `make logs`. If it says `trainer.ami_id is empty`,
run `make ami` then `make apply`. If it says `no capacity in any of N AZs`, that is g6e
spot being genuinely scarce — it keeps retrying with backoff.

**Trainer boots and dies immediately.** `make trainer-logs`, or read
`/var/log/llm/trainer.log` over SSH. The most likely causes are a deploy key that is still
the Terraform placeholder, or a `code_sha` that does not exist.

**No progress.json.** The syncer failed. `sudo systemctl status llm-checkpoint-syncer` on
the trainer. The orchestrator will terminate and relaunch after `stall_minutes`.

**Run finished but no DONE marker.** The trainer only writes `DONE` when the highest local
checkpoint reaches `Config().steps`. Check the actual step with `make status`.

**Everything is stuck.** `make stop` terminates the trainer; the orchestrator launches a
fresh one on the next tick.
