module "observability" {
  source = "../../modules/observability"

  project            = var.project
  alert_email        = var.alert_email
  monthly_budget_usd = var.monthly_budget_usd
  log_retention_days = var.log_retention_days
}
