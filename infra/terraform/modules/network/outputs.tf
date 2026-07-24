output "vpc_id" {
  description = "Project VPC."
  value       = aws_vpc.main.id
}

output "subnet_ids" {
  description = "Subnet id per AZ, passed to the orchestrator for RunInstances."
  value       = { for az, s in aws_subnet.public : az => s.id }
}

output "trainer_azs" {
  description = "AZs that offer the trainer instance type. The orchestrator rotates through these on capacity errors."
  value       = local.trainer_azs
}

output "trainer_sg_id" {
  description = "Security group for the trainer instance."
  value       = aws_security_group.trainer.id
}

output "orchestrator_sg_id" {
  description = "Security group for the orchestrator instance."
  value       = aws_security_group.orchestrator.id
}
