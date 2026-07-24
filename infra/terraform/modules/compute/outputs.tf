output "project" {
  description = "Project name, used as the llm:project tag filter in Makefile queries."
  value       = var.project
}

output "run_id" {
  description = "Current run identifier."
  value       = var.run_id
}

output "run_prefix" {
  description = "S3 URI prefixes for the current run_id."
  value = {
    checkpoints = "s3://${var.bucket_id}/checkpoints/${var.run_id}/"
    final       = "s3://${var.bucket_id}/final/${var.run_id}/"
    logs        = "s3://${var.bucket_id}/logs/${var.run_id}/"
    state       = "s3://${var.bucket_id}/state/${var.run_id}/"
  }
}

output "orchestrator_public_ip" {
  description = "Elastic IP of the orchestrator. Stable across reboots."
  value       = try(aws_eip.orchestrator[0].public_ip, null)
}

output "orchestrator_ssh" {
  description = "Ready-to-run SSH command for the orchestrator."
  value = try(
    "ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${aws_eip.orchestrator[0].public_ip}",
    "orchestrator not deployed (set orchestrator_ami_id)",
  )
}
