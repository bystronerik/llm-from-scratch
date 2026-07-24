module "ssm" {
  source = "../../modules/ssm"

  project                   = var.project
  deploy_key_parameter_name = var.deploy_key_parameter_name
  git_repo_ssh_url          = var.git_repo_ssh_url
}
