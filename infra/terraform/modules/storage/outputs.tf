output "bucket_id" {
  description = "S3 bucket holding checkpoints, finals, logs and run state."
  value       = aws_s3_bucket.artifacts.id
}

output "bucket_arn" {
  description = "ARN of the artifacts bucket, for IAM policies."
  value       = aws_s3_bucket.artifacts.arn
}
