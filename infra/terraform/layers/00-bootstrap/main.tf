# Creates the S3 bucket that holds the main stack's Terraform state.
#
# Run once, before anything else. Uses LOCAL state on purpose -- a bucket cannot store
# the state that describes its own creation without a chicken-and-egg problem. The
# resulting terraform.tfstate here is small and static; losing it is recoverable by
# importing the bucket.
#
#   cd infra/terraform/bootstrap
#   terraform init
#   terraform apply
#   terraform output -raw state_bucket    # feed to `make init`

terraform {
  required_version = ">= 1.11"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.0, < 7.0"
    }
  }
}

variable "aws_region" {
  description = "Region for the state bucket. Must match the `region` in the main stack's backend block."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named AWS profile. Empty string uses the default credential chain."
  type        = string
  default     = "default"
}

variable "project" {
  description = "Name prefix, matching the main stack."
  type        = string
  default     = "llm-training"
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile != "" ? var.aws_profile : null

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
      Component = "bootstrap"
    }
  }
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "state" {
  bucket = "${var.project}-tfstate-${data.aws_caller_identity.current.account_id}"

  # State buckets should outlive a careless `terraform destroy` in this directory.
  lifecycle {
    prevent_destroy = true
  }
}

# Versioning is the recovery path for a corrupted or truncated state write. Not optional.
resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Terraform state is plaintext and contains every attribute of every resource. Deny any
# non-TLS access outright rather than relying on clients to opt in.
resource "aws_s3_bucket_policy" "state" {
  bucket = aws_s3_bucket.state.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource = [
        aws_s3_bucket.state.arn,
        "${aws_s3_bucket.state.arn}/*",
      ]
      Condition = {
        Bool = { "aws:SecureTransport" = "false" }
      }
    }]
  })

  depends_on = [aws_s3_bucket_public_access_block.state]
}

# Old state versions pile up fast. Keep 90 days of history, then let them go.
resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    id     = "expire-old-state-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.state]
}

output "state_bucket" {
  description = "Pass to the main stack: terraform init -backend-config=\"bucket=<this>\""
  value       = aws_s3_bucket.state.id
}
