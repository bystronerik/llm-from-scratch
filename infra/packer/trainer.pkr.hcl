# Base is the AWS Deep Learning Base AMI: NVIDIA driver and CUDA are already installed and
# known-good together. Building from plain Ubuntu would add ~15 minutes and a whole class
# of driver/CUDA mismatch failures for no benefit.
data "amazon-ami" "dlami" {
  filters = {
    name                = var.trainer_base_ami_name
    virtualization-type = "hvm"
    root-device-type    = "ebs"
    architecture        = "x86_64"
  }
  most_recent = true
  owners      = [var.trainer_base_ami_owner]
  region      = var.aws_region
  profile     = var.aws_profile != "" ? var.aws_profile : null
}

source "amazon-ebs" "trainer" {
  ami_name        = "${var.project}-trainer-${local.timestamp}"
  ami_description = "LLM trainer: CUDA torch venv, corpus and token cache pre-staged"

  source_ami = data.amazon-ami.dlami.id
  region     = var.aws_region
  profile    = var.aws_profile != "" ? var.aws_profile : null

  ssh_username = "ubuntu"
  ssh_timeout  = "20m"

  # Build on spot: a ~15 minute build costs cents, and building on the real instance type
  # means the CUDA smoke test validates the exact stack that will train.
  spot_price                  = "auto"
  spot_instance_types         = [var.trainer_build_instance_type]
  temporary_key_pair_type     = "ed25519"
  associate_public_ip_address = true

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = var.trainer_root_volume_gb
    volume_type           = "gp3"
    delete_on_termination = true
  }

  # IMDSv2 only, matching the runtime posture.
  imds_support = "v2.0"
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  tags = {
    Name      = "${var.project}-trainer"
    Project   = var.project
    ManagedBy = "packer"
    Component = "trainer"
    BaseAMI   = "{{ .SourceAMI }}"
  }

  snapshot_tags = {
    Name    = "${var.project}-trainer"
    Project = var.project
  }
}

build {
  name    = "trainer"
  sources = ["source.amazon-ebs.trainer"]

  provisioner "shell" {
    inline = ["mkdir -p /tmp/runtime"]
  }

  provisioner "file" {
    source      = "${path.root}/../runtime/trainer/"
    destination = "/tmp/runtime/"
  }

  provisioner "shell" {
    script = "${path.root}/scripts/trainer-provision.sh"

    environment_vars = [
      "HF_TOKEN=${var.hf_token}",
      "GIT_REPO=${var.git_repo_https_url}",
      "CODE_SHA=${var.code_sha}",
      "NUM_ARTICLES=${var.num_articles}",
      "DEBIAN_FRONTEND=noninteractive",
    ]

    # sudo -E so HF_TOKEN survives into the provisioning shell.
    execute_command = "chmod +x {{ .Path }}; sudo -E env {{ .Vars }} {{ .Path }}"
  }

  post-processor "manifest" {
    output     = "${path.root}/manifest-trainer.json"
    strip_path = true
    custom_data = {
      component = "trainer"
    }
  }
}
