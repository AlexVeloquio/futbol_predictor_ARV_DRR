terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.aws_region }

variable "aws_region"    { default = "us-east-1" }
variable "project_name"  { default = "futbol-predictor" }

# S3 — Model artifacts
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models"
}

# ECR — Docker images
resource "aws_ecr_repository" "inference" {
  name         = "${var.project_name}-inference"
  force_delete = true
}

# DynamoDB — Prediction cache
resource "aws_dynamodb_table" "predictions" {
  name         = "${var.project_name}-predictions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "match_key"
  attribute { name = "match_key"; type = "S" }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda" {
  name = "${var.project_name}-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow",
      Principal = { Service = "lambda.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy" "lambda" {
  name = "${var.project_name}-policy"
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.models.arn}/*" },
      { Effect = "Allow", Action = ["dynamodb:PutItem", "dynamodb:GetItem"], Resource = aws_dynamodb_table.predictions.arn },
      { Effect = "Allow", Action = ["logs:*"], Resource = "arn:aws:logs:*:*:*" },
    ]
  })
}

# Lambda — Inference
resource "aws_lambda_function" "inference" {
  function_name = "${var.project_name}-inference"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.inference.repository_url}:latest"
  timeout       = 30
  memory_size   = 512
  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.models.id
      DYNAMO_TABLE = aws_dynamodb_table.predictions.name
    }
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "api" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "predict" {
  api_id             = aws_apigatewayv2_api.api.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.inference.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.predict.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "api_gw" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

output "api_endpoint" { value = aws_apigatewayv2_api.api.api_endpoint }
output "s3_bucket"    { value = aws_s3_bucket.models.id }
