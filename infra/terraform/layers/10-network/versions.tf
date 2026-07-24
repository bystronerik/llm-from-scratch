terraform {
  # 1.11 is the floor for S3 backend `use_lockfile`, which replaced the old DynamoDB lock
  # table.
  required_version = ">= 1.11"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # Floor is 6.0, not 5.x: this config reads `data.aws_region.current.region`, an
      # attribute that only exists in provider v6 (v5 exposes `.name`).
      version = ">= 6.0, < 7.0"
    }
  }

  # Partial config: the state bucket is created by layers/00-bootstrap, so it cannot be
  # referenced here (backends take no variables). `make init` supplies it:
  #
  #   terraform init \
  #     -backend-config="bucket=<state-bucket-from-bootstrap>" \
  #     -backend-config="profile=<your-aws-profile>"
  backend "s3" {
    key          = "llm-training/network/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}
