# Upstream layer outputs. The Makefile applies layers in order (10 -> 60), so by the time
# this layer runs, all of these states exist.

data "terraform_remote_state" "network" {
  backend = "s3"
  config = {
    bucket  = var.state_bucket
    key     = "llm-training/network/terraform.tfstate"
    region  = "us-east-1"
    profile = var.aws_profile != "" ? var.aws_profile : null
  }
}

data "terraform_remote_state" "storage" {
  backend = "s3"
  config = {
    bucket  = var.state_bucket
    key     = "llm-training/storage/terraform.tfstate"
    region  = "us-east-1"
    profile = var.aws_profile != "" ? var.aws_profile : null
  }
}

data "terraform_remote_state" "ssm" {
  backend = "s3"
  config = {
    bucket  = var.state_bucket
    key     = "llm-training/ssm/terraform.tfstate"
    region  = "us-east-1"
    profile = var.aws_profile != "" ? var.aws_profile : null
  }
}

data "terraform_remote_state" "observability" {
  backend = "s3"
  config = {
    bucket  = var.state_bucket
    key     = "llm-training/observability/terraform.tfstate"
    region  = "us-east-1"
    profile = var.aws_profile != "" ? var.aws_profile : null
  }
}

data "terraform_remote_state" "iam" {
  backend = "s3"
  config = {
    bucket  = var.state_bucket
    key     = "llm-training/iam/terraform.tfstate"
    region  = "us-east-1"
    profile = var.aws_profile != "" ? var.aws_profile : null
  }
}
