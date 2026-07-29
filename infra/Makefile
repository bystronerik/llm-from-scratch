# Thin wrappers around terraform, packer and the AWS CLI.
#
#   make bootstrap   once, creates the Terraform state bucket
#   make init        wires the S3 backend of every layer to that bucket
#   make deploy-key  pushes the GitHub deploy key to SSM (never via Terraform)
#   make apply       creates/updates every layer in dependency order
#   make ami         builds both AMIs
#   make status      what is running right now
#   make logs        follow the orchestrator
#
# The stack is split into independent state layers under terraform/layers/, applied in
# order (10-network -> 60-compute). `make apply` walks them; `make apply-one LAYER=<name>`
# drives a single layer.
#
# Fastest path from nothing to a running job: bootstrap, init, apply, deploy-key,
# ami, apply.

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

TF_DIR      := terraform
LAYERS_DIR  := $(TF_DIR)/layers
BOOTSTRAP   := $(LAYERS_DIR)/00-bootstrap
PACKER_DIR  := packer
TESTS_DIR   := tests

# Layers in apply order. destroy walks the reverse.
LAYERS          := 10-network 20-storage 30-ssm 40-observability 50-iam
LAYERS_REVERSED := 50-iam 40-observability 30-ssm 20-storage 10-network

# Layers whose terraform_remote_state data sources need the state bucket passed in.
STATE_LAYERS := 50-iam 60-compute

# Shared values live in common.tfvars; each layer auto-loads its own terraform.tfvars.
COMMON_VARFILE := ../../common.tfvars

# Shorthands for reading a specific layer's outputs.
STORAGE := terraform -chdir=$(LAYERS_DIR)/20-storage
SSM     := terraform -chdir=$(LAYERS_DIR)/30-ssm
COMPUTE := terraform -chdir=$(LAYERS_DIR)/60-compute

# Read shared config from common.tfvars so the Makefile never duplicates it. Commented or
# missing values fall back to the defaults baked into the layer variables.
# PROFILE may legitimately be empty (default credential chain); REGION/PROJECT fall back to
# the same defaults the layer variables carry.
PROFILE := $(shell grep -E '^[[:space:]]*aws_profile[[:space:]]*=' $(TF_DIR)/common.tfvars 2>/dev/null | sed 's/.*=[[:space:]]*"\(.*\)"/\1/')
REGION  := $(or $(shell grep -E '^[[:space:]]*aws_region[[:space:]]*='  $(TF_DIR)/common.tfvars 2>/dev/null | sed 's/.*=[[:space:]]*"\(.*\)"/\1/'),us-east-1)
PROJECT := $(or $(shell grep -E '^[[:space:]]*project[[:space:]]*=' $(TF_DIR)/common.tfvars 2>/dev/null | sed 's/.*=[[:space:]]*"\(.*\)"/\1/'),llm-training)
AWS     := aws --region $(REGION) $(if $(PROFILE),--profile $(PROFILE),)

# Used only to discover which AZs can host the trainer AMI build. The fleet's actual
# candidate types live in packer/variables.pkr.hcl.
TRAINER_TYPE := g6e.xlarge

# The state bucket, read back from bootstrap. Empty until `make bootstrap` has run.
state_bucket = $(shell terraform -chdir=$(BOOTSTRAP) output -raw state_bucket 2>/dev/null)

# AMI ids come from the Packer manifests, so `make apply` after `make ami` just works.
TRAINER_AMI      := $(shell jq -r '.builds[-1].artifact_id // ""' $(PACKER_DIR)/manifest-trainer.json 2>/dev/null | cut -d: -f2)
ORCHESTRATOR_AMI := $(shell jq -r '.builds[-1].artifact_id // ""' $(PACKER_DIR)/manifest-orchestrator.json 2>/dev/null | cut -d: -f2)
TF_AMI_VARS      := $(if $(TRAINER_AMI),-var trainer_ami_id=$(TRAINER_AMI),) \
                    $(if $(ORCHESTRATOR_AMI),-var orchestrator_ami_id=$(ORCHESTRATOR_AMI),)

.PHONY: help
help:
	@echo "Setup (in order):"
	@echo "  bootstrap     Create the Terraform state bucket (once per account)"
	@echo "  init          Initialise every layer's backend against that bucket"
	@echo "  keypair       Create an EC2 key pair for SSH access"
	@echo "  deploy-key    Push a GitHub deploy key to SSM      KEY=~/.ssh/llm_deploy"
	@echo "  ami           Build both AMIs (trainer + orchestrator)"
	@echo ""
	@echo "Operate:"
	@echo "  plan / apply / destroy          all layers, in (reverse) dependency order"
	@echo "  plan-one / apply-one LAYER=<n>  a single layer (e.g. LAYER=60-compute)"
	@echo "  status        Instances, progress and run state"
	@echo "  logs          Follow the orchestrator's journal"
	@echo "  trainer-logs  Tail the current trainer's log"
	@echo "  ssh           SSH to the orchestrator"
	@echo "  checkpoints   List checkpoints in S3"
	@echo "  fetch-final   Download the finished model"
	@echo "  stop          Terminate the trainer (orchestrator will relaunch)"
	@echo ""
	@echo "Checks:"
	@echo "  test / validate / fmt"
	@echo ""
	@echo "  layers=$(LAYERS)"
	@echo "  region=$(REGION) profile=$(PROFILE)"
	@echo "  trainer_ami=$(or $(TRAINER_AMI),<not built>) orchestrator_ami=$(or $(ORCHESTRATOR_AMI),<not built>)"

# --- Setup ------------------------------------------------------------------

.PHONY: bootstrap
bootstrap:
	terraform -chdir=$(BOOTSTRAP) init
	terraform -chdir=$(BOOTSTRAP) apply
	@echo ""
	@echo "State bucket: $$(terraform -chdir=$(BOOTSTRAP) output -raw state_bucket)"
	@echo "Now run: make init"

.PHONY: init
init:
	@bucket=$(state_bucket) ; \
	if [[ -z $$bucket ]]; then echo "No state bucket. Run 'make bootstrap' first." >&2; exit 1; fi ; \
	echo "Backend bucket: $$bucket" ; \
	for L in $(LAYERS); do \
	  echo "== init $$L ==" ; \
	  terraform -chdir=$(LAYERS_DIR)/$$L init -reconfigure \
	    -backend-config="bucket=$$bucket" \
	    $(if $(PROFILE),-backend-config="profile=$(PROFILE)",) ; \
	done

.PHONY: keypair
keypair:
	@name=$${NAME:-llm-training} ; \
	if $(AWS) ec2 describe-key-pairs --key-names "$$name" >/dev/null 2>&1; then \
	  echo "Key pair '$$name' already exists." ; \
	else \
	  $(AWS) ec2 create-key-pair --key-name "$$name" \
	    --query KeyMaterial --output text > "$$HOME/.ssh/$$name.pem" ; \
	  chmod 400 "$$HOME/.ssh/$$name.pem" ; \
	  echo "Created ~/.ssh/$$name.pem" ; \
	fi ; \
	echo "Set ssh_key_name = \"$$name\" in $(LAYERS_DIR)/60-compute/terraform.tfvars"

# Deliberately NOT a Terraform resource: Terraform state is plaintext, so a private key
# passed through a variable would be readable by anyone with access to the state bucket.
.PHONY: deploy-key
deploy-key:
	@if [[ -z "$(KEY)" ]]; then \
	  echo "Usage: make deploy-key KEY=~/.ssh/llm_deploy_key" >&2 ; \
	  echo "" >&2 ; \
	  echo "Generate one with:" >&2 ; \
	  echo "  ssh-keygen -t ed25519 -f ~/.ssh/llm_deploy_key -N '' -C 'llm-training'" >&2 ; \
	  echo "then add ~/.ssh/llm_deploy_key.pub as a READ-ONLY deploy key on the GitHub repo." >&2 ; \
	  exit 1 ; \
	fi
	@param=$$($(SSM) output -raw deploy_key_parameter) ; \
	$(AWS) ssm put-parameter --name "$$param" --type SecureString \
	  --value "file://$(KEY)" --overwrite >/dev/null ; \
	echo "Wrote $(KEY) to SSM parameter $$param"

# --- AMIs -------------------------------------------------------------------

.PHONY: ami ami-trainer ami-orchestrator
ami: ami-trainer ami-orchestrator

# A Packer EC2 instance request covers one subnet, so the build lives or dies with a single
# AZ -- and g6e capacity is patchy per AZ. Rotate through the default subnet of every AZ
# that offers the type and stop at the first with capacity, the same strategy the
# orchestrator uses at runtime. Anything that is not a capacity error aborts immediately:
# another AZ will not fix it. AZ=us-east-1c pins the build to one zone.
#
# $${PIPESTATUS[0]} rather than `set -o pipefail`: .SHELLFLAGS above needs make >= 3.82 and
# macOS ships 3.81, so tee's exit status would otherwise mask every failed build.
ami-trainer:
	@if [[ -z "$${HF_TOKEN:-}" ]] && [[ -f ../.env ]]; then \
	  echo "Using HF_TOKEN from ../.env" ; \
	fi
	cd $(PACKER_DIR) && packer init .
	@azs="$(AZ)" ; \
	if [[ -z $$azs ]]; then \
	  azs=$$($(AWS) ec2 describe-instance-type-offerings --location-type availability-zone \
	    --filters "Name=instance-type,Values=$(TRAINER_TYPE)" \
	    --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort) ; \
	fi ; \
	if [[ -z $$azs ]]; then echo "No AZ in $(REGION) offers $(TRAINER_TYPE)." >&2 ; exit 1 ; fi ; \
	log=$$(mktemp) ; trap 'rm -f $$log' EXIT ; built= ; \
	for az in $$azs; do \
	  subnet=$$($(AWS) ec2 describe-subnets \
	    --filters "Name=default-for-az,Values=true" "Name=availability-zone,Values=$$az" \
	    --query 'Subnets[0].SubnetId' --output text) ; \
	  if [[ -z $$subnet || $$subnet == None ]]; then \
	    echo "No default subnet in $$az, skipping." >&2 ; continue ; \
	  fi ; \
	  echo "== building trainer AMI in $$az ($$subnet) ==" ; \
	  (cd $(PACKER_DIR) && packer build \
	     -var project=$(PROJECT) \
	     -var aws_region=$(REGION) \
	     $(if $(PROFILE),-var aws_profile=$(PROFILE),) \
	     -var hf_token="$${HF_TOKEN:-$$(grep -E '^HF_TOKEN=' ../../.env 2>/dev/null | cut -d= -f2-)}" \
	     $(if $(CODE_SHA),-var code_sha=$(CODE_SHA),) \
	     -var trainer_build_subnet_id=$$subnet \
	     -only='trainer.*' .) 2>&1 | tee "$$log" ; \
	  rc=$${PIPESTATUS[0]} ; \
	  if [[ $$rc -eq 0 ]]; then built=1 ; break ; fi ; \
	  if ! grep -qiE 'InsufficientInstanceCapacity|sufficient .* capacity|capacity-not-available' "$$log" ; then \
	    echo "Build failed in $$az for a non-capacity reason; not rotating." >&2 ; exit 1 ; \
	  fi ; \
	  echo "No $(TRAINER_TYPE) capacity in $$az, trying the next AZ." >&2 ; \
	done ; \
	if [[ -z $$built ]]; then \
	  echo "No capacity in any AZ that offers $(TRAINER_TYPE). Retry later." >&2 ; exit 1 ; \
	fi
	@echo "Trainer AMI: $$(jq -r '.builds[-1].artifact_id' $(PACKER_DIR)/manifest-trainer.json | cut -d: -f2)"

ami-orchestrator:
	cd $(PACKER_DIR) && packer init . && packer build \
	  -var aws_region=$(REGION) \
	  $(if $(PROFILE),-var aws_profile=$(PROFILE),) \
	  -only='orchestrator.*' .
	@echo "Orchestrator AMI: $$(jq -r '.builds[-1].artifact_id' $(PACKER_DIR)/manifest-orchestrator.json | cut -d: -f2)"

# --- Terraform --------------------------------------------------------------
#
# The state bucket is injected into the two layers whose terraform_remote_state data
# sources read upstream outputs; the AMI ids only reach the compute layer.

# $(1) = terraform subcommand (plan|apply|destroy), $(2) = layer name. Expects $$bucket set.
define tf_layer
	extra="" ; \
	case " $(STATE_LAYERS) " in *" $(2) "*) extra="-var state_bucket=$$bucket" ;; esac ; \
	case "$(2)" in 60-compute) extra="$$extra $(TF_AMI_VARS)" ;; esac ; \
	echo "== $(1) $(2) ==" ; \
	terraform -chdir=$(LAYERS_DIR)/$(2) $(1) -var-file=$(COMMON_VARFILE) $$extra
endef

.PHONY: plan apply
plan apply:
	@bucket=$(state_bucket) ; \
	if [[ -z $$bucket ]]; then echo "No state bucket. Run 'make bootstrap && make init' first." >&2; exit 1; fi ; \
	for L in $(LAYERS); do $(call tf_layer,$@,$$L) ; done

.PHONY: plan-one apply-one
plan-one: OP := plan
apply-one: OP := apply
plan-one apply-one:
	@test -n "$(LAYER)" || { echo "Usage: make $@ LAYER=60-compute" >&2; exit 1; }
	@bucket=$(state_bucket) ; \
	if [[ -z $$bucket ]]; then echo "No state bucket. Run 'make bootstrap && make init' first." >&2; exit 1; fi ; \
	$(call tf_layer,$(OP),$(LAYER))

.PHONY: destroy
destroy:
	@echo "This destroys the VPC, orchestrator and IAM roles across every layer."
	@echo "The S3 artifact bucket (storage layer) holds your checkpoints -- Terraform will"
	@echo "refuse to delete it while it has objects. Empty it manually if that is really"
	@echo "intended."
	@bucket=$(state_bucket) ; \
	for L in $(LAYERS_REVERSED); do $(call tf_layer,destroy,$$L) ; done

# --- Operate ----------------------------------------------------------------

.PHONY: status
status:
	@echo "=== instances ==="
	@$(AWS) ec2 describe-instances \
	  --filters "Name=tag:llm:project,Values=$(PROJECT)" \
	            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
	  --query 'Reservations[].Instances[].{ID:InstanceId,Role:Tags[?Key==`llm:role`]|[0].Value,Type:InstanceType,State:State.Name,AZ:Placement.AvailabilityZone,IP:PublicIpAddress}' \
	  --output table 2>/dev/null || echo "  (none)"
	@echo "=== run state ==="
	@run=$$($(COMPUTE) output -json run_prefix 2>/dev/null | jq -r '.state') ; \
	if [[ -z $$run || $$run == null ]]; then echo "  (compute layer not applied)" ; else \
	  $(AWS) s3 ls "$$run" 2>/dev/null || echo "  (no state yet)" ; \
	  echo "=== progress ===" ; \
	  $(AWS) s3 cp "$${run}progress.json" - 2>/dev/null | jq . || echo "  (no progress yet)" ; \
	fi

.PHONY: logs
logs:
	@ip=$$($(COMPUTE) output -raw orchestrator_public_ip) ; \
	key=$$(grep -E '^[[:space:]]*ssh_key_name' $(LAYERS_DIR)/60-compute/terraform.tfvars 2>/dev/null | sed 's/.*=[[:space:]]*"\(.*\)"/\1/') ; \
	ssh -i "$$HOME/.ssh/$$key.pem" -o StrictHostKeyChecking=accept-new ubuntu@$$ip \
	  'sudo journalctl -u llm-orchestrator -f --no-pager'

.PHONY: trainer-logs
trainer-logs:
	@key=$$(grep -E '^[[:space:]]*ssh_key_name' $(LAYERS_DIR)/60-compute/terraform.tfvars 2>/dev/null | sed 's/.*=[[:space:]]*"\(.*\)"/\1/') ; \
	ip=$$($(AWS) ec2 describe-instances \
	  --filters "Name=tag:llm:role,Values=trainer" "Name=instance-state-name,Values=running" \
	  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text) ; \
	if [[ $$ip == "None" || -z $$ip ]]; then echo "No trainer running." >&2; exit 1; fi ; \
	ssh -i "$$HOME/.ssh/$$key.pem" -o StrictHostKeyChecking=accept-new ubuntu@$$ip \
	  'sudo tail -f /var/log/llm/trainer.log'

.PHONY: ssh
ssh:
	@$(COMPUTE) output -raw orchestrator_ssh

.PHONY: checkpoints
checkpoints:
	@$(AWS) s3 ls "$$($(COMPUTE) output -json run_prefix | jq -r '.checkpoints')" --human-readable

.PHONY: fetch-final
fetch-final:
	@dest=$${DEST:-../.cache/final} ; mkdir -p "$$dest" ; \
	$(AWS) s3 sync "$$($(COMPUTE) output -json run_prefix | jq -r '.final')" "$$dest" ; \
	echo "Downloaded to $$dest"

# Terminates the trainer. The orchestrator treats this as an interruption and launches a
# replacement, so use it to force a restart -- not to stop the run.
.PHONY: stop
stop:
	@id=$$($(AWS) ec2 describe-instances \
	  --filters "Name=tag:llm:role,Values=trainer" "Name=instance-state-name,Values=running" \
	  --query 'Reservations[0].Instances[0].InstanceId' --output text) ; \
	if [[ $$id == "None" || -z $$id ]]; then echo "No trainer running."; exit 0; fi ; \
	echo "Terminating $$id (the orchestrator will relaunch it)" ; \
	$(AWS) ec2 terminate-instances --instance-ids "$$id" --output table

# --- Checks -----------------------------------------------------------------

.PHONY: test validate fmt
test:
	@echo "=== orchestrator logic ==="
	@python3 $(TESTS_DIR)/test_orchestrator.py | tail -3
	@echo "=== shell syntax ==="
	@for f in runtime/trainer/*.sh $(PACKER_DIR)/scripts/*.sh; do bash -n "$$f" && echo "  ok $$f"; done
	@echo "=== python syntax ==="
	@python3 -m py_compile runtime/orchestrator/orchestrator.py && echo "  ok orchestrator.py"

# validate does not need the backend or the state bucket: init -backend=false installs the
# providers and records the modules, which is all `terraform validate` requires.
validate:
	@for L in $(LAYERS); do \
	  echo "== validate $$L ==" ; \
	  terraform -chdir=$(LAYERS_DIR)/$$L init -backend=false -input=false >/dev/null ; \
	  terraform -chdir=$(LAYERS_DIR)/$$L validate ; \
	done
	@echo "== validate 00-bootstrap ==" ; \
	terraform -chdir=$(BOOTSTRAP) init -backend=false -input=false >/dev/null ; \
	terraform -chdir=$(BOOTSTRAP) validate
	cd $(PACKER_DIR) && packer validate -syntax-only .

fmt:
	terraform -chdir=$(TF_DIR) fmt -recursive
	cd $(PACKER_DIR) && packer fmt .
