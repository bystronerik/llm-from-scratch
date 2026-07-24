output "trainer_instance_profile_name" {
  description = "Instance profile the orchestrator attaches to each trainer it launches."
  value       = module.iam.trainer_instance_profile_name
}

output "orchestrator_instance_profile_name" {
  description = "Instance profile attached to the orchestrator instance."
  value       = module.iam.orchestrator_instance_profile_name
}

output "trainer_role_arn" {
  description = "Trainer role ARN."
  value       = module.iam.trainer_role_arn
}
