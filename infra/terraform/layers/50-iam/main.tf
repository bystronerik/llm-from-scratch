module "iam" {
  source = "../../modules/iam"

  project               = var.project
  trainer_instance_type = var.trainer_instance_type

  # Cross-layer inputs from remote state.
  bucket_arn           = data.terraform_remote_state.storage.outputs.bucket_arn
  deploy_key_param_arn = data.terraform_remote_state.ssm.outputs.deploy_key_param_arn
  log_group_arn        = data.terraform_remote_state.observability.outputs.log_group_arn
  trainer_sg_id        = data.terraform_remote_state.network.outputs.trainer_sg_id
  subnet_ids           = data.terraform_remote_state.network.outputs.subnet_ids
}
