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

# --- IAM --------------------------------------------------------------------

variable "trainer_instance_type" {
  description = <<-EOT
    Instance type the orchestrator is allowed to launch (RunInstances condition).

    Shared with the network and compute layers; keep the three in sync if you override it.
  EOT
  type        = string
  default     = "g6e.xlarge"
}

# --- Remote state -----------------------------------------------------------

variable "state_bucket" {
  description = <<-EOT
    S3 bucket holding Terraform state, created by layers/00-bootstrap. Read by the
    terraform_remote_state data sources below to pull the network/storage/ssm/observability
    outputs. Injected by the Makefile from the bootstrap output.
  EOT
  type        = string
}
