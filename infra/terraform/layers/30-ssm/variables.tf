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

# --- Secrets ----------------------------------------------------------------

variable "deploy_key_parameter_name" {
  description = <<-EOT
    SSM Parameter Store path holding the read-only GitHub deploy key.

    Created with a placeholder value and lifecycle.ignore_changes; the real key is written
    out-of-band by `make deploy-key` so the private key never lands in Terraform state.
  EOT
  type        = string
  default     = "/llm-training/github-deploy-key"
}

variable "git_repo_ssh_url" {
  description = "Repository the deploy key grants access to. Used only in the description."
  type        = string
  default     = "git@github.com:bystronerik/llm-from-scratch.git"
}
