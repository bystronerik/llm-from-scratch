provider "aws" {
  region = var.aws_region

  # Empty string means "use the default credential chain" (env vars, SSO, instance role).
  profile = var.aws_profile != "" ? var.aws_profile : null

  default_tags {
    tags = merge(
      {
        Project   = var.project
        ManagedBy = "terraform"
        Repo      = "bystronerik/llm-from-scratch"
      },
      var.extra_tags,
    )
  }
}
