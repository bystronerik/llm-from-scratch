variable "project" {
  description = "Name prefix; used in the parameter's Name tag."
  type        = string
}

variable "deploy_key_parameter_name" {
  description = "SSM Parameter Store path holding the read-only GitHub deploy key."
  type        = string
}

variable "git_repo_ssh_url" {
  description = "Repository the deploy key grants access to. Used only in the description."
  type        = string
}
