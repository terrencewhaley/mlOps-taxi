# MLOps Taxi — Interview Spin-Up Runbook

## Spin Up (run before interview)

# 1. Delete stale IAM service account stack if it exists

aws cloudformation delete-stack \
 --stack-name eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa \
 --region us-east-1

# Wait 30 seconds for deletion to complete, then:

# 2. Create node group

eksctl create nodegroup \
 --cluster mlops-taxi \
 --region us-east-1 \
 --name standard-workers \
 --node-type t3.small \
 --nodes 2 \
 --nodes-min 1 \
 --nodes-max 3 \
 --managed

# 3. Associate OIDC provider

eksctl utils associate-iam-oidc-provider \
 --cluster mlops-taxi \
 --region us-east-1 \
 --approve

# 4. Create service account

eksctl create iamserviceaccount \
 --cluster mlops-taxi \
 --region us-east-1 \
 --name mlops-taxi-sa \
 --namespace default \
 --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
 --approve

# 5. Update kubeconfig

aws eks update-kubeconfig --region us-east-1 --name mlops-taxi

# 6. Deploy

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 7. Get new load balancer URL (wait 2-3 min)

kubectl get services

# 8. Update vercel.json with new load balancer URL and push

git add . && git commit -m "Update load balancer URL" && git push

## Teardown

# 1. Delete node group first

eksctl delete nodegroup \
 --cluster mlops-taxi \
 --region us-east-1 \
 --name standard-workers

# 2. Wait for completion, then verify

kubectl get nodes
