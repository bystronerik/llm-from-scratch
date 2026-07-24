output "vpc_id" {
  description = "Project VPC."
  value       = module.network.vpc_id
}

output "subnet_ids" {
  description = "Subnet id per AZ."
  value       = module.network.subnet_ids
}

output "trainer_azs" {
  description = "AZs that offer the trainer instance type."
  value       = module.network.trainer_azs
}

output "trainer_sg_id" {
  description = "Trainer security group id."
  value       = module.network.trainer_sg_id
}

output "orchestrator_sg_id" {
  description = "Orchestrator security group id."
  value       = module.network.orchestrator_sg_id
}
