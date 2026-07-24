# The supervisor. A t4g.small (~$12/month) that outlives every trainer and is the only
# thing that decides when a new spot instance is launched.
#
# On-demand, not spot: it must survive precisely the capacity events it exists to handle.
# Count-gated on the AMI id so `terraform apply` works before any Packer build has run.

data "aws_region" "current" {}

locals {
  orchestrator_enabled = var.orchestrator_ami_id != "" ? 1 : 0

  # Everything the daemon needs, rendered to /etc/llm/orchestrator.json at boot. JSON
  # rather than an env file because the subnet map is structured data.
  orchestrator_config = {
    region       = data.aws_region.current.region
    project      = var.project
    run_id       = var.run_id
    bucket       = var.bucket_id
    log_group    = var.log_group_name
    poll_seconds = var.poll_interval_seconds

    # Stop conditions
    max_run_hours  = var.max_run_hours
    max_relaunches = var.max_relaunches
    stall_minutes  = var.stall_minutes

    # How to build a trainer
    trainer = {
      ami_id           = var.trainer_ami_id
      instance_type    = var.trainer_instance_type
      instance_profile = var.trainer_instance_profile_name
      security_group   = var.trainer_sg_id
      key_name         = var.ssh_key_name
      root_volume_gb   = var.trainer_root_volume_gb
      # AZ -> subnet. The orchestrator rotates through these on capacity errors.
      subnets = var.subnet_ids
    }

    # Passed through into the trainer's /etc/llm/run.env
    run_env = {
      RUN_ID           = var.run_id
      S3_BUCKET        = var.bucket_id
      CODE_SHA         = var.code_sha
      GIT_REPO         = var.git_repo_ssh_url
      DEPLOY_KEY_PARAM = var.deploy_key_param_name
      LOG_GROUP        = var.log_group_name
      AWS_REGION       = data.aws_region.current.region
      PROJECT          = var.project
      # No sample generation on an unattended box: keeps the 2-minute spot interruption
      # budget spent on saving the checkpoint. See src/utils.py:pipeline_mode().
      PIPELINE = "true"
    }
  }
}

resource "aws_instance" "orchestrator" {
  count = local.orchestrator_enabled

  ami                    = var.orchestrator_ami_id
  instance_type          = var.orchestrator_instance_type
  subnet_id              = var.subnet_ids[var.trainer_azs[0]]
  vpc_security_group_ids = [var.orchestrator_sg_id]
  iam_instance_profile   = var.orchestrator_instance_profile_name
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  root_block_device {
    volume_size           = var.orchestrator_root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_endpoint               = "enabled"
    http_put_response_hop_limit = 1
  }

  user_data = templatefile("${path.module}/templates/orchestrator-user-data.sh.tftpl", {
    config_json = jsonencode(local.orchestrator_config)
  })

  # Changing run config should redeploy the supervisor, not silently do nothing.
  user_data_replace_on_change = true

  tags = {
    Name          = "${var.project}-orchestrator"
    "llm:project" = var.project
    "llm:run-id"  = var.run_id
    "llm:role"    = "orchestrator"
  }
}

# Stable SSH target across reboots and redeploys. Free while attached to a running
# instance; billed only if left allocated with nothing attached. The internet gateway it
# needs is created by the network layer, which this layer's remote-state dependency on the
# network outputs guarantees is applied first.
resource "aws_eip" "orchestrator" {
  count = local.orchestrator_enabled

  instance = aws_instance.orchestrator[0].id
  domain   = "vpc"

  tags = { Name = "${var.project}-orchestrator" }
}
