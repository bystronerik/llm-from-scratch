output "deploy_key_parameter" {
  description = "SSM parameter the trainer reads its GitHub deploy key from. Populate with `make deploy-key`."
  value       = module.ssm.param_name
}

output "deploy_key_param_arn" {
  description = "ARN of the deploy-key parameter, for IAM policies."
  value       = module.ssm.param_arn
}
