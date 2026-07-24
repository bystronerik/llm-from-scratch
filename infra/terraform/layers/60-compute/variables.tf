# --- Provider (shared; set in ../../common.tfvars) --------------------------

variable "project" {
  description = "Name prefix for every resource, and the value of the Project tag."
  type        = string
  default     = "llm-training"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,30}$", var.project))
    error_message = "project must be lowercase alphanumeric with hyphens, 3-31 chars (it becomes part of an S3 bucket name)."
  }
}

variable "aws_region" {
  description = "Region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named profile from ~/.aws/credentials. Empty string uses the default credential chain."
  type        = string
  default     = "default"
}

variable "extra_tags" {
  description = "Additional tags merged into the provider's default_tags for every resource."
  type        = map(string)
  default     = {}
}

# --- Run configuration ------------------------------------------------------

variable "run_id" {
  description = <<-EOT
    Identifier for one logical training run. Namespaces everything under the S3 prefixes and
    tags the trainer so the orchestrator can find it. Change to start fresh; keep to resume.
  EOT
  type        = string
  default     = "run-001"

  validation {
    condition     = can(regex("^[a-zA-Z0-9._-]{1,64}$", var.run_id))
    error_message = "run_id must be 1-64 chars of alphanumerics, dot, underscore or hyphen (it becomes an S3 prefix and a tag value)."
  }
}

variable "code_sha" {
  description = "Git commit the trainer checks out. Pin to a full SHA for a real run; \"main\" is for smoke tests."
  type        = string
  default     = "main"
}

variable "max_run_hours" {
  description = "Hard wall-clock cap. After this the orchestrator stops relaunching and writes an ABORTED marker."
  type        = number
  default     = 48

  validation {
    condition     = var.max_run_hours > 0 && var.max_run_hours <= 720
    error_message = "max_run_hours must be between 0 and 720 (30 days)."
  }
}

variable "poll_interval_seconds" {
  description = "How often the orchestrator checks trainer state and the S3 DONE marker."
  type        = number
  default     = 30
}

variable "stall_minutes" {
  description = "Relaunch the trainer if state/progress.json has not advanced in this long. Must exceed the checkpoint interval."
  type        = number
  default     = 25
}

variable "max_relaunches" {
  description = "Give up after this many spot relaunches. 0 disables the cap."
  type        = number
  default     = 50
}

# --- Instances --------------------------------------------------------------

variable "orchestrator_instance_type" {
  description = "Supervisor instance. arm64/Graviton -- must match the orchestrator AMI's architecture."
  type        = string
  default     = "t4g.small"
}

variable "trainer_instance_type" {
  description = <<-EOT
    GPU instance for training, launched as one-time spot.

    Shared with the network and iam layers; keep the three in sync if you override it.
  EOT
  type        = string
  default     = "g6e.xlarge"
}

variable "trainer_ami_id" {
  description = "Trainer AMI (x86_64) from `make ami-trainer`. Empty skips the compute resources; the Makefile passes it as -var."
  type        = string
  default     = ""
}

variable "orchestrator_ami_id" {
  description = "Orchestrator AMI (arm64) from `make ami-orchestrator`. Empty skips the compute resources; the Makefile passes it as -var."
  type        = string
  default     = ""
}

variable "trainer_root_volume_gb" {
  description = "Root EBS volume for the trainer (OS, CUDA venv, baked corpus). Checkpoints go to instance-store NVMe."
  type        = number
  default     = 120
}

variable "orchestrator_root_volume_gb" {
  description = "Root EBS volume for the orchestrator. It only runs a Python daemon."
  type        = number
  default     = 20
}

variable "ssh_key_name" {
  description = "Name of an EXISTING EC2 key pair. Empty launches instances with no key pair. Create one with `make keypair`."
  type        = string
  default     = ""
}

variable "git_repo_ssh_url" {
  description = "Repository the trainer clones at boot. Shared with the ssm layer."
  type        = string
  default     = "git@github.com:bystronerik/llm-from-scratch.git"
}

# --- Remote state -----------------------------------------------------------

variable "state_bucket" {
  description = <<-EOT
    S3 bucket holding Terraform state, created by layers/00-bootstrap. Read by the
    terraform_remote_state data sources to pull the upstream layers' outputs. Injected by
    the Makefile from the bootstrap output.
  EOT
  type        = string
}
