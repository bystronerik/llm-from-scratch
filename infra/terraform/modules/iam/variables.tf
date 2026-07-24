variable "project" {
  description = "Name prefix, and the llm:project tag value the terminate fence checks."
  type        = string
}

variable "trainer_instance_type" {
  description = "Instance type the orchestrator is allowed to launch (RunInstances condition)."
  type        = string
}

# --- Cross-layer inputs (fed from remote state) -----------------------------

variable "bucket_arn" {
  description = "ARN of the artifacts bucket (from the storage layer)."
  type        = string
}

variable "deploy_key_param_arn" {
  description = "ARN of the deploy-key SSM parameter (from the ssm layer)."
  type        = string
}

variable "log_group_arn" {
  description = "ARN of the shared CloudWatch log group (from the observability layer)."
  type        = string
}

variable "trainer_sg_id" {
  description = "Trainer security group id the orchestrator may launch into (from the network layer)."
  type        = string
}

variable "subnet_ids" {
  description = "AZ -> subnet id map the orchestrator may launch into (from the network layer)."
  type        = map(string)
}
