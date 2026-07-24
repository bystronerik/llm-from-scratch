# Cost and log-retention guardrails for the whole system.
#
# The account-wide monthly budget with email alerts at 80% and 100%, plus the single
# CloudWatch log group that both instances write to.

# --- Budget -----------------------------------------------------------------
#
# Deliberately NOT filtered to this project's tag. Tag-based budget filters require the
# tag to first be activated as a cost allocation tag in Billing -- a manual console step
# with a ~24h propagation delay. A budget that silently matches nothing is worse than no
# budget, so this watches the whole account. On a personal learning account that is also
# the more useful signal: it catches the resource you forgot about, not just this stack.
#
# Note that AWS Budgets and its notifications are always billed in us-east-1 regardless of
# var.aws_region; that is a service constraint, not a configuration choice.

resource "aws_budgets_budget" "monthly" {
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Warning shot with time left to react.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  # FORECASTED fires when AWS projects the month will end over budget -- usually days
  # before ACTUAL crosses 100%, which is the whole point of having it.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

# --- Logs -------------------------------------------------------------------
#
# Shared by both instances. Retention is finite on purpose: these logs are for debugging a
# run in flight, not a permanent record. Anything worth keeping is archived to logs/ in S3.
resource "aws_cloudwatch_log_group" "main" {
  name              = "/${var.project}"
  retention_in_days = var.log_retention_days
}
