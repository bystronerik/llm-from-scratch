output "log_group_name" {
  description = "CloudWatch log group carrying both instances' output."
  value       = module.observability.log_group_name
}

output "log_group_arn" {
  description = "ARN of the shared log group, for IAM policies."
  value       = module.observability.log_group_arn
}
