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
  description = "Region to deploy into. us-east-1 has the best g6e spot capacity and price."
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

# --- Network ----------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR for the project VPC. Public subnets are carved out per AZ."
  type        = string
  default     = "10.42.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "vpc_cidr must be a valid CIDR block."
  }
}

variable "ssh_allowed_cidr" {
  description = <<-EOT
    CIDR allowed to reach port 22 on both instances.

    PLACEHOLDER: 203.0.113.0/32 is TEST-NET-3 (RFC 5737), reserved for documentation and
    not routable -- leaving it means nobody can SSH in. Replace with "<your.ip>/32"; find
    it with: curl -s https://checkip.amazonaws.com
  EOT
  type        = string
  default     = "203.0.113.0/32"

  validation {
    condition     = can(cidrhost(var.ssh_allowed_cidr, 0))
    error_message = "ssh_allowed_cidr must be a valid CIDR block, e.g. \"203.0.113.4/32\"."
  }
}

variable "trainer_instance_type" {
  description = <<-EOT
    GPU instance for training. Determines which AZs can host a subnet.

    Shared with the iam and compute layers: if you override it, set the same value in all
    three (or the RunInstances fence and the launched type will disagree).
  EOT
  type        = string
  default     = "g6e.xlarge"
}
