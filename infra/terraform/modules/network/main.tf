# VPC with one public subnet per AZ that can actually run the trainer instance type.
#
# No NAT gateway: it would cost ~$32/month and buy nothing here. Instances get public IPs
# for egress (GitHub, PyPI) and inbound is closed by security groups. S3 traffic bypasses
# the internet gateway entirely via the gateway endpoint below.

data "aws_region" "current" {}

# AZs the account can actually use. Availability zone *names* are per-account aliases, so
# this must be looked up rather than hardcoded.
data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

# The AZs that offer g6e.xlarge. This is the reason subnets are built dynamically: newer
# GPU types are not offered in every AZ (us-east-1e notably lags), and a hardcoded list
# silently rots as AWS changes its footprint. Querying it means the orchestrator's AZ
# rotation can only ever target AZs that could really serve the request.
data "aws_ec2_instance_type_offerings" "trainer" {
  filter {
    name   = "instance-type"
    values = [var.trainer_instance_type]
  }

  location_type = "availability-zone"
}

locals {
  # Intersect so we never create a subnet in an AZ that is unusable for another reason.
  trainer_azs = sort(tolist(setintersection(
    toset(data.aws_ec2_instance_type_offerings.trainer.locations),
    toset(data.aws_availability_zones.available.names),
  )))

  # Stable index per AZ so subnet CIDRs do not shuffle when AWS adds a new AZ to the list.
  az_index = { for i, az in local.trainer_azs : az => i }

  name_prefix = var.project
}

# Fail loudly at plan time rather than producing a VPC with zero usable subnets.
resource "terraform_data" "trainer_az_guard" {
  lifecycle {
    precondition {
      condition     = length(local.trainer_azs) > 0
      error_message = "No availability zone in ${data.aws_region.current.region} offers ${var.trainer_instance_type}. Check the instance type name, or pick another region."
    }
  }
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = local.name_prefix }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = local.name_prefix }
}

resource "aws_subnet" "public" {
  for_each = toset(local.trainer_azs)

  vpc_id                  = aws_vpc.main.id
  availability_zone       = each.key
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, local.az_index[each.key])
  map_public_ip_on_launch = true

  tags = { Name = "${local.name_prefix}-public-${each.key}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${local.name_prefix}-public" }
}

resource "aws_route_table_association" "public" {
  for_each = aws_subnet.public

  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

# Free, and keeps 254 MB checkpoint uploads on the AWS private network instead of routing
# through the internet gateway. Same-region EC2->S3 transfer is not billed either way, so
# this is for security and reliability rather than cost.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${data.aws_region.current.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.public.id]

  tags = { Name = "${local.name_prefix}-s3" }
}

# --- Security groups --------------------------------------------------------
#
# Two groups rather than one shared: the trainer can be locked down further later without
# touching your own access to the orchestrator.

resource "aws_security_group" "trainer" {
  name_prefix = "${local.name_prefix}-trainer-"
  description = "Training instance: SSH from the operator only, unrestricted egress."
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name_prefix}-trainer" }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "orchestrator" {
  name_prefix = "${local.name_prefix}-orchestrator-"
  description = "Orchestrator instance: SSH from the operator only, unrestricted egress."
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name_prefix}-orchestrator" }

  lifecycle {
    create_before_destroy = true
  }
}

# Rules as separate resources so changing the allowed CIDR does not force the whole
# security group (and therefore the instances using it) to be replaced.
resource "aws_vpc_security_group_ingress_rule" "trainer_ssh" {
  security_group_id = aws_security_group.trainer.id
  description       = "SSH from the operator"
  cidr_ipv4         = var.ssh_allowed_cidr
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "trainer_all" {
  security_group_id = aws_security_group.trainer.id
  description       = "GitHub, PyPI, S3, SSM"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "orchestrator_ssh" {
  security_group_id = aws_security_group.orchestrator.id
  description       = "SSH from the operator"
  cidr_ipv4         = var.ssh_allowed_cidr
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "orchestrator_all" {
  security_group_id = aws_security_group.orchestrator.id
  description       = "EC2 API, S3, SSM"
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
