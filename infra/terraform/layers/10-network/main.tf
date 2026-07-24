module "network" {
  source = "../../modules/network"

  project               = var.project
  vpc_cidr              = var.vpc_cidr
  ssh_allowed_cidr      = var.ssh_allowed_cidr
  trainer_instance_type = var.trainer_instance_type
}
