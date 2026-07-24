variable "project" {
  description = "Name prefix for network resources."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR for the project VPC. Public subnets are carved out per AZ."
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR allowed to reach port 22 on both instances."
  type        = string
}

variable "trainer_instance_type" {
  description = "GPU instance type. Determines which AZs can host a subnet."
  type        = string
}
