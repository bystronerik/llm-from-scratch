# Two roles, each scoped to what its instance actually does.
#
# The trainer gets NO EC2 permissions at all -- it cannot launch or terminate anything,
# including itself (it shuts down through the OS). The orchestrator gets EC2 lifecycle
# permissions but is fenced in by conditions so a bug in the supervision loop cannot
# launch the wrong instance type or terminate something it did not create.

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  bucket_arn = var.bucket_arn

  # Prefixes the trainer is allowed to touch. Deliberately not the whole bucket.
  trainer_object_arns = [
    "${local.bucket_arn}/checkpoints/*",
    "${local.bucket_arn}/final/*",
    "${local.bucket_arn}/logs/*",
    "${local.bucket_arn}/state/*",
  ]
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# --- Trainer ----------------------------------------------------------------

resource "aws_iam_role" "trainer" {
  name_prefix        = "${var.project}-trainer-"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "trainer" {
  statement {
    sid       = "ReadWriteRunArtifacts"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = local.trainer_object_arns
  }

  # ListBucket is a bucket-level action, so it cannot carry an object prefix in Resource.
  # Restrict it with a condition instead, or the trainer could enumerate the whole bucket.
  statement {
    sid       = "ListRunPrefixes"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["checkpoints/*", "final/*", "logs/*", "state/*", ""]
    }
  }

  statement {
    sid       = "ReadDeployKey"
    actions   = ["ssm:GetParameter"]
    resources = [var.deploy_key_param_arn]
  }

  # SecureString parameters are encrypted with the account's default SSM key; reading one
  # requires Decrypt on that key, scoped by the SSM service context.
  statement {
    sid       = "DecryptDeployKey"
    actions   = ["kms:Decrypt"]
    resources = ["arn:aws:kms:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:key/*"]

    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${data.aws_region.current.region}.amazonaws.com"]
    }
  }

  statement {
    sid       = "WriteLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${var.log_group_arn}:*"]
  }
}

resource "aws_iam_role_policy" "trainer" {
  name_prefix = "trainer-"
  role        = aws_iam_role.trainer.id
  policy      = data.aws_iam_policy_document.trainer.json
}

resource "aws_iam_instance_profile" "trainer" {
  name_prefix = "${var.project}-trainer-"
  role        = aws_iam_role.trainer.name
}

# --- Orchestrator -----------------------------------------------------------

resource "aws_iam_role" "orchestrator" {
  name_prefix        = "${var.project}-orchestrator-"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

data "aws_iam_policy_document" "orchestrator" {
  # Describe* does not support resource-level permissions -- "*" is the only legal value.
  # This is read-only metadata, so the blast radius is disclosure, not mutation.
  statement {
    sid = "InspectFleet"
    actions = [
      "ec2:DescribeInstances",
      "ec2:DescribeInstanceStatus",
      "ec2:DescribeInstanceTypeOfferings",
      "ec2:DescribeSpotPriceHistory",
      "ec2:DescribeImages",
      "ec2:DescribeSubnets",
      "ec2:DescribeSecurityGroups",
      "ec2:DescribeTags",
    ]
    resources = ["*"]
  }

  # RunInstances authorizes against every resource it touches. Split into two statements so
  # the instance-type condition lands on the instance resource, where it is evaluated --
  # attaching it to the whole call would silently deny everything, since conditions on
  # ec2:InstanceType do not apply to subnets or images.
  statement {
    sid       = "LaunchTrainerInstance"
    actions   = ["ec2:RunInstances"]
    resources = ["arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "ec2:InstanceType"
      values   = [var.trainer_instance_type]
    }
  }

  statement {
    sid     = "LaunchTrainerSupportingResources"
    actions = ["ec2:RunInstances"]
    resources = concat(
      [
        "arn:aws:ec2:${data.aws_region.current.region}::image/*",
        "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:volume/*",
        "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:network-interface/*",
        "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:key-pair/*",
        "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:security-group/${var.trainer_sg_id}",
        "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:spot-instances-request/*",
      ],
      [for id in values(var.subnet_ids) : "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:subnet/${id}"],
    )
  }

  # Tagging is only permitted as part of a launch, not as a standalone call -- so the
  # orchestrator cannot retag an existing instance to bring it under its own control and
  # then terminate it.
  statement {
    sid     = "TagOnLaunch"
    actions = ["ec2:CreateTags"]
    resources = [
      "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:instance/*",
      "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:volume/*",
      "arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:spot-instances-request/*",
    ]

    condition {
      test     = "StringEquals"
      variable = "ec2:CreateAction"
      values   = ["RunInstances"]
    }
  }

  # The important fence: terminate only instances this orchestrator tagged as its own.
  statement {
    sid       = "TerminateOwnTrainers"
    actions   = ["ec2:TerminateInstances"]
    resources = ["arn:aws:ec2:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "ec2:ResourceTag/llm:managed-by"
      values   = ["orchestrator"]
    }

    condition {
      test     = "StringEquals"
      variable = "ec2:ResourceTag/llm:project"
      values   = [var.project]
    }
  }

  # Required to launch an instance carrying the trainer's instance profile. Scoped to
  # exactly that one role -- a wildcard here is the classic IAM privilege-escalation hole,
  # since it would let the orchestrator launch an instance as any role in the account.
  statement {
    sid       = "PassTrainerRoleOnly"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.trainer.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ec2.amazonaws.com"]
    }
  }

  statement {
    sid     = "ReadWriteRunState"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      "${local.bucket_arn}/state/*",
      "${local.bucket_arn}/logs/*",
      "${local.bucket_arn}/checkpoints/*",
      "${local.bucket_arn}/final/*",
    ]
  }

  statement {
    sid       = "ListBucketForProgress"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arn]
  }

  statement {
    sid       = "WriteLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${var.log_group_arn}:*"]
  }
}

resource "aws_iam_role_policy" "orchestrator" {
  name_prefix = "orchestrator-"
  role        = aws_iam_role.orchestrator.id
  policy      = data.aws_iam_policy_document.orchestrator.json
}

resource "aws_iam_instance_profile" "orchestrator" {
  name_prefix = "${var.project}-orchestrator-"
  role        = aws_iam_role.orchestrator.name
}
