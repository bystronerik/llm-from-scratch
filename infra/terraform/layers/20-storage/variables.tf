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

# --- Storage retention ------------------------------------------------------

variable "checkpoint_retention_days" {
  description = <<-EOT
    Days before intermediate checkpoints under checkpoints/ are expired. The syncer already
    prunes live, so this is the backstop for abandoned runs. Finals are never expired.
  EOT
  type        = number
  default     = 30

  validation {
    condition     = var.checkpoint_retention_days >= 1
    error_message = "checkpoint_retention_days must be at least 1."
  }
}

variable "log_retention_days" {
  description = <<-EOT
    Retention for archived logs under logs/. Shared with the observability layer (which
    uses it for the CloudWatch log group); keep the two in sync if you override it.
  EOT
  type        = number
  default     = 30

  validation {
    condition = contains(
      [1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653],
      var.log_retention_days,
    )
    error_message = "log_retention_days must be one of the values CloudWatch Logs accepts (1, 3, 5, 7, 14, 30, 60, 90, 120, ... 3653)."
  }
}
