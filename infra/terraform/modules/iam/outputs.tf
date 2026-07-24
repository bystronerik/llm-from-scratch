output "trainer_instance_profile_name" {
  description = "Instance profile the orchestrator attaches to each trainer it launches."
  value       = aws_iam_instance_profile.trainer.name
}

output "orchestrator_instance_profile_name" {
  description = "Instance profile attached to the orchestrator instance."
  value       = aws_iam_instance_profile.orchestrator.name
}

output "trainer_role_arn" {
  description = "Trainer role ARN (the orchestrator is scoped to PassRole exactly this)."
  value       = aws_iam_role.trainer.arn
}
