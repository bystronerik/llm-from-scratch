module "compute" {
  source = "../../modules/compute"

  project                     = var.project
  run_id                      = var.run_id
  code_sha                    = var.code_sha
  max_run_hours               = var.max_run_hours
  poll_interval_seconds       = var.poll_interval_seconds
  stall_minutes               = var.stall_minutes
  max_relaunches              = var.max_relaunches
  orchestrator_instance_type  = var.orchestrator_instance_type
  trainer_instance_type       = var.trainer_instance_type
  trainer_ami_id              = var.trainer_ami_id
  orchestrator_ami_id         = var.orchestrator_ami_id
  trainer_root_volume_gb      = var.trainer_root_volume_gb
  orchestrator_root_volume_gb = var.orchestrator_root_volume_gb
  ssh_key_name                = var.ssh_key_name
  git_repo_ssh_url            = var.git_repo_ssh_url

  # Cross-layer inputs from remote state.
  bucket_id                          = data.terraform_remote_state.storage.outputs.bucket_id
  log_group_name                     = data.terraform_remote_state.observability.outputs.log_group_name
  deploy_key_param_name              = data.terraform_remote_state.ssm.outputs.deploy_key_parameter
  trainer_instance_profile_name      = data.terraform_remote_state.iam.outputs.trainer_instance_profile_name
  orchestrator_instance_profile_name = data.terraform_remote_state.iam.outputs.orchestrator_instance_profile_name
  trainer_sg_id                      = data.terraform_remote_state.network.outputs.trainer_sg_id
  orchestrator_sg_id                 = data.terraform_remote_state.network.outputs.orchestrator_sg_id
  subnet_ids                         = data.terraform_remote_state.network.outputs.subnet_ids
  trainer_azs                        = data.terraform_remote_state.network.outputs.trainer_azs
}
