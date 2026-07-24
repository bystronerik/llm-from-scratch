locals {
  timestamp = formatdate("YYYYMMDD-hhmmss", timestamp())
}

variable "project" {
  type        = string
  default     = "llm-training"
  description = "Name prefix for AMIs and tags. Must match the Terraform variable."
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "aws_profile" {
  type        = string
  default     = ""
  description = "Named AWS profile. Empty uses the default credential chain."
}

# --- Trainer AMI ------------------------------------------------------------

variable "trainer_base_ami_name" {
  type        = string
  default     = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"
  description = <<-EOT
    Name filter for the base DLAMI.

    VERIFY before the first build -- AWS renames these periodically:
      aws ec2 describe-images --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia*Ubuntu*" \
        --query 'reverse(sort_by(Images,&CreationDate))[:5].[Name,ImageId]' --output table
  EOT
}

variable "trainer_base_ami_owner" {
  type        = string
  default     = "898082745236"
  description = "AWS account that publishes the Deep Learning AMIs."
}

variable "trainer_build_instance_type" {
  type        = string
  default     = "g6e.xlarge"
  description = "Build on the real training type so the CUDA smoke test is meaningful."
}

variable "trainer_root_volume_gb" {
  type        = number
  default     = 120
  description = "OS + CUDA torch venv + uv cache (~15 GB) + staged data (~0.9 GB), with headroom."
}

# --- Orchestrator AMI -------------------------------------------------------

variable "orchestrator_base_ami_name" {
  type        = string
  default     = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"
  description = "Ubuntu 24.04 LTS arm64. Note arm64 -- the orchestrator is Graviton."
}

variable "orchestrator_base_ami_owner" {
  type        = string
  default     = "099720109477"
  description = "Canonical's AWS account id."
}

variable "orchestrator_build_instance_type" {
  type    = string
  default = "t4g.small"
}

# --- Content ----------------------------------------------------------------

variable "git_repo_https_url" {
  type        = string
  default     = "https://github.com/bystronerik/llm-from-scratch.git"
  description = <<-EOT
    HTTPS clone URL, used ONLY at build time to warm the uv cache and generate the token
    cache. If the repo is private, pass a token-bearing URL via -var, or pre-clone.

    The runtime clone uses SSH with the deploy key from SSM; this is a separate path.
  EOT
}

variable "code_sha" {
  type        = string
  default     = "main"
  description = "Commit used to resolve dependencies and build the token cache at bake time."
}

variable "num_articles" {
  type        = number
  default     = 100000
  description = "Must match Config.num_articles -- it is part of the corpus and token-cache filenames."
}

variable "hf_token" {
  type        = string
  default     = ""
  sensitive   = true
  description = <<-EOT
    HuggingFace token, used ONLY during the build to download the Wikipedia dataset.

    It is passed as an environment variable to the provisioning script and never written
    to disk, so it does not end up in the AMI. The 19 GB raw dataset is deleted after the
    470 MB corpus is derived from it.
  EOT
}
