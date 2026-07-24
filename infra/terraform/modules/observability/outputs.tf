output "log_group_name" {
  description = "CloudWatch log group carrying both instances' output."
  value       = aws_cloudwatch_log_group.main.name
}

output "log_group_arn" {
  description = "ARN of the shared log group, for IAM policies."
  value       = aws_cloudwatch_log_group.main.arn
}
