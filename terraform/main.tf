terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "docker-desktop"
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