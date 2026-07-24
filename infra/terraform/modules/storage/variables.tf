variable "project" {
  description = "Name prefix; part of the bucket name."
  type        = string
}

variable "checkpoint_retention_days" {
  description = "Days before intermediate checkpoints under checkpoints/ are expired."
  type        = number
}

variable "log_retention_days" {
  description = "Days before archived logs under logs/ are expired."
  type        = number
}
