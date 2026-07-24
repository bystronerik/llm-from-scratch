variable "project" {
  description = "Name prefix and llm:project tag value."
  type        = string
}

# --- Run configuration ------------------------------------------------------

variable "run_id" {
  description = "Identifier for one logical training run."
  type        = string
}

variable "code_sha" {
  description = "Git commit the trainer checks out."
  type        = string
}

variable "max_run_hours" {
  description = "Hard wall-clock cap before the orchestrator gives up."
  type        = number
}

variable "poll_interval_seconds" {
  description = "How often the orchestrator checks trainer state and the S3 DONE marker."
  type        = number
}

variable "stall_minutes" {
  description = "Relaunch the trainer if progress.json has not advanced in this long."
  type        = number
}

variable "max_relaunches" {
  description = "Give up after this many spot relaunches. 0 disables the cap."
  type        = number
}

# --- Instances --------------------------------------------------------------

variable "orchestrator_instance_type" {
  description = "Supervisor instance type (arm64/Graviton)."
  type        = string
}

variable "trainer_instance_type" {
  description = "GPU instance type the orchestrator launches as spot."
  type        = string
}

variable "trainer_ami_id" {
  description = "Trainer AMI (x86_64). Empty means compute resources are skipped."
  type        = string
}

variable "orchestrator_ami_id" {
  description = "Orchestrator AMI (arm64). Empty means compute resources are skipped."
  type        = string
}

variable "trainer_root_volume_gb" {
  description = "Root EBS volume for the trainer."
  type        = number
}

variable "orchestrator_root_volume_gb" {
  description = "Root EBS volume for the orchestrator."
  type        = number
}

variable "ssh_key_name" {
  description = "Existing EC2 key pair attached to both instances. Empty launches without a key."
  type        = string
}

variable "git_repo_ssh_url" {
  description = "Repository the trainer clones at boot."
  type        = string
}

# --- Cross-layer inputs (fed from remote state) -----------------------------

variable "bucket_id" {
  description = "Artifacts bucket name (from the storage layer)."
  type        = string
}

variable "log_group_name" {
  description = "Shared CloudWatch log group name (from the observability layer)."
  type        = string
}

variable "deploy_key_param_name" {
  description = "Deploy-key SSM parameter name (from the ssm layer)."
  type        = string
}

variable "trainer_instance_profile_name" {
  description = "Instance profile the orchestrator attaches to trainers (from the iam layer)."
  type        = string
}

variable "orchestrator_instance_profile_name" {
  description = "Instance profile for the orchestrator instance (from the iam layer)."
  type        = string
}

variable "trainer_sg_id" {
  description = "Trainer security group id (from the network layer)."
  type        = string
}

variable "orchestrator_sg_id" {
  description = "Orchestrator security group id (from the network layer)."
  type        = string
}

variable "subnet_ids" {
  description = "AZ -> subnet id map (from the network layer)."
  type        = map(string)
}

variable "trainer_azs" {
  description = "Ordered AZs that offer the trainer type (from the network layer)."
  type        = list(string)
}
