output "bucket_id" {
  description = "S3 bucket holding checkpoints, finals, logs and run state."
  value       = module.storage.bucket_id
}

output "bucket_arn" {
  description = "ARN of the artifacts bucket, for IAM policies."
  value       = module.storage.bucket_arn
}
