# Read-only GitHub deploy key, used by the trainer to clone the private repo at boot.
#
# Terraform creates the parameter with a placeholder and then deliberately stops managing
# its value. The real key is written out-of-band by `make deploy-key`.
#
# Why: Terraform state stores every attribute in PLAINTEXT, including SecureString values
# after they are read back. Putting the private key in a .tfvars file or a variable would
# copy it into the state bucket, where anyone with s3:GetObject can read it. Writing it
# with `aws ssm put-parameter` keeps it out of state entirely.

resource "aws_ssm_parameter" "deploy_key" {
  name        = var.deploy_key_parameter_name
  description = "Read-only GitHub deploy key for ${var.git_repo_ssh_url}"
  type        = "SecureString"

  # Placeholder. `make deploy-key` overwrites it; the trainer fails fast and loudly if it
  # still sees this value, rather than hanging on an SSH prompt.
  value = "PLACEHOLDER-run-make-deploy-key"

  lifecycle {
    ignore_changes = [value]
  }

  tags = { Name = "${var.project}-deploy-key" }
}
