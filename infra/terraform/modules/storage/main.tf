# The single durable store for the whole system.
#
# Layout is top-level-prefix-first, NOT run-id-first:
#
#   checkpoints/<run_id>/step_<N>.pt      temporary, synced from instance-store scratch
#   final/<run_id>/...                    final checkpoint, results.csv, sample output
#   logs/<run_id>/...                     archived training logs
#   state/<run_id>/progress.json          {step, timestamp, instance_id} -- stall detection
#   state/<run_id>/DONE                   completion marker; stops the orchestrator
#   state/<run_id>/ABORTED                wall-clock or relaunch cap was hit
#   state/<run_id>/interrupted-<ts>.json  one breadcrumb per spot reclaim
#
# Grouping by run id first would read more naturally, but S3 lifecycle filters match a
# LITERAL prefix -- there are no wildcards, so `runs/*/checkpoints/` is inexpressible.
# Putting the category first is what lets "expire checkpoints, keep finals forever" be a
# declarative rule instead of a cron job.

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project}-${data.aws_caller_identity.current.account_id}-${data.aws_region.current.region}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*",
      ]
      Condition = {
        Bool = { "aws:SecureTransport" = "false" }
      }
    }]
  })

  depends_on = [aws_s3_bucket_public_access_block.artifacts]
}

# Exactly one lifecycle configuration per bucket -- a second resource does not merge, it
# replaces. All rules belong here.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  # A spot instance dying mid-upload leaves multipart parts that are billed but do not
  # show up in the object listing. This is the only thing that cleans them up.
  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 3
    }
  }

  # Intermediate checkpoints are disposable once a run finishes. The syncer prunes them
  # live; this is the backstop for runs that were abandoned rather than completed.
  rule {
    id     = "expire-intermediate-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    expiration {
      days = var.checkpoint_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }

  rule {
    id     = "expire-archived-logs"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    expiration {
      days = var.log_retention_days
    }
  }

  # Finals are the point of the whole exercise. Never expire them; just move them to
  # cheaper storage once they stop being touched.
  rule {
    id     = "archive-final-artifacts"
    status = "Enabled"

    filter {
      prefix = "final/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  # state/ holds markers and progress.json -- bytes, not megabytes. No expiry; the run
  # history is worth keeping.

  depends_on = [aws_s3_bucket_versioning.artifacts]
}
