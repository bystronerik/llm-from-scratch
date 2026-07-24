output "param_name" {
  description = "SSM parameter the trainer reads its GitHub deploy key from. Populate with `make deploy-key`."
  value       = aws_ssm_parameter.deploy_key.name
}

output "param_arn" {
  description = "ARN of the deploy-key parameter, for IAM policies."
  value       = aws_ssm_parameter.deploy_key.arn
}
