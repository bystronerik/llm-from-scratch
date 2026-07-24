output "project" {
  description = "Project name, used as the llm:project tag filter in Makefile queries."
  value       = module.compute.project
}

output "run_id" {
  description = "Current run identifier."
  value       = module.compute.run_id
}

output "run_prefix" {
  description = "S3 URI prefixes for the current run_id."
  value       = module.compute.run_prefix
}

output "orchestrator_public_ip" {
  description = "Elastic IP of the orchestrator. Stable across reboots."
  value       = module.compute.orchestrator_public_ip
}

output "orchestrator_ssh" {
  description = "Ready-to-run SSH command for the orchestrator."
  value       = module.compute.orchestrator_ssh
}
