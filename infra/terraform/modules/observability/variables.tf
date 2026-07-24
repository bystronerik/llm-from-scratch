variable "project" {
  description = "Name prefix; part of the budget name and log group path."
  type        = string
}

variable "alert_email" {
  description = "Address that receives AWS Budgets notifications."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget. Notifications fire at 80% and 100%."
  type        = number
}

variable "log_retention_days" {
  description = "Retention for the shared CloudWatch log group."
  type        = number
}
