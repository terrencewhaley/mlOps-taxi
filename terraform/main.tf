terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "docker-desktop"
}

provider "aws" {
  region = "us-east-1"
}

# --- Look up the VPC eksctl already created for the mlops-taxi cluster ---
data "aws_vpc" "eks_vpc" {
  tags = {
    "alpha.eksctl.io/cluster-name" = "mlops-taxi"
  }
}

data "aws_subnets" "eks_private_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.eks_vpc.id]
  }
  tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# --- Security group: only allow Redis traffic (port 6379) from inside the VPC ---
resource "aws_security_group" "redis_sg" {
  name        = "mlops-taxi-redis-sg"
  description = "Allow Redis access from EKS pods"
  vpc_id      = data.aws_vpc.eks_vpc.id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.eks_vpc.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlops-taxi-redis-sg"
  }
}

# --- Subnet group: tells ElastiCache which subnets it may launch into ---
resource "aws_elasticache_subnet_group" "redis_subnet_group" {
  name       = "mlops-taxi-redis-subnet-group"
  subnet_ids = data.aws_subnets.eks_private_subnets.ids
}

# --- The Redis cluster itself ---
resource "aws_elasticache_cluster" "redis" {
  cluster_id         = "mlops-taxi-redis"
  engine             = "redis"
  engine_version     = "7.1"
  node_type          = "cache.t3.micro"
  num_cache_nodes    = 1
  port               = 6379
  subnet_group_name  = aws_elasticache_subnet_group.redis_subnet_group.name
  security_group_ids = [aws_security_group.redis_sg.id]
  apply_immediately  = true
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}

resource "kubernetes_deployment" "mlops_taxi" {
  metadata {
    name = "mlops-taxi-tf"
    labels = {
      app = "mlops-taxi-tf"
    }
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = "mlops-taxi-tf"
      }
    }

    template {
      metadata {
        labels = {
          app = "mlops-taxi-tf"
        }
      }

      spec {
        container {
          name              = "mlops-taxi"
          image             = "mlops-taxi:latest"
          image_pull_policy = "Never"

          port {
            container_port = 8000
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "mlops_taxi" {
  metadata {
    name = "mlops-taxi-tf-service"
  }

  spec {
    selector = {
      app = "mlops-taxi-tf"
    }

    port {
      port        = 8080
      target_port = 8000
    }

    type = "LoadBalancer"
  }
}