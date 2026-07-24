# Plain Ubuntu arm64 -- the orchestrator only runs a boto3 daemon, so there is nothing to
# gain from a specialised base image.
data "amazon-ami" "ubuntu_arm64" {
  filters = {
    name                = var.orchestrator_base_ami_name
    virtualization-type = "hvm"
    root-device-type    = "ebs"
    architecture        = "arm64"
  }
  most_recent = true
  owners      = [var.orchestrator_base_ami_owner]
  region      = var.aws_region
  profile     = var.aws_profile != "" ? var.aws_profile : null
}

source "amazon-ebs" "orchestrator" {
  ami_name        = "${var.project}-orchestrator-${local.timestamp}"
  ami_description = "LLM training orchestrator: boto3 supervision loop"

  source_ami    = data.amazon-ami.ubuntu_arm64.id
  instance_type = var.orchestrator_build_instance_type
  region        = var.aws_region
  profile       = var.aws_profile != "" ? var.aws_profile : null

  ssh_username = "ubuntu"
  ssh_timeout  = "15m"

  associate_public_ip_address = true
  temporary_key_pair_type     = "ed25519"

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 20
    volume_type           = "gp3"
    delete_on_termination = true
  }

  imds_support = "v2.0"

  tags = {
    Name      = "${var.project}-orchestrator"
    Project   = var.project
    ManagedBy = "packer"
    Component = "orchestrator"
    BaseAMI   = "{{ .SourceAMI }}"
  }

  snapshot_tags = {
    Name    = "${var.project}-orchestrator"
    Project = var.project
  }
}

build {
  name    = "orchestrator"
  sources = ["source.amazon-ebs.orchestrator"]

  provisioner "shell" {
    inline = ["mkdir -p /tmp/runtime"]
  }

  provisioner "file" {
    source      = "${path.root}/../runtime/orchestrator/"
    destination = "/tmp/runtime/"
  }

  provisioner "shell" {
    script           = "${path.root}/scripts/orchestrator-provision.sh"
    environment_vars = ["DEBIAN_FRONTEND=noninteractive"]
    execute_command  = "chmod +x {{ .Path }}; sudo -E env {{ .Vars }} {{ .Path }}"
  }

  post-processor "manifest" {
    output     = "${path.root}/manifest-orchestrator.json"
    strip_path = true
    custom_data = {
      component = "orchestrator"
    }
  }
}
