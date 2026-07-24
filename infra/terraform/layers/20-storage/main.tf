module "storage" {
  source = "../../modules/storage"

  project                   = var.project
  checkpoint_retention_days = var.checkpoint_retention_days
  log_retention_days        = var.log_retention_days
}
