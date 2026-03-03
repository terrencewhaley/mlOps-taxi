# MLOps Taxi — Interview Spin-Up Runbook

## Spin Up (run before interview)

1. Create node group (15-20 min):
   eksctl create nodegroup \
    --cluster mlops-taxi \
    --region us-east-1 \
    --name standard-workers \
    --node-type t3.small \
    --nodes 2 \
    --nodes-min 1 \
    --nodes-max 3 \
    --managed

2. Associate OIDC provider:
   eksctl utils associate-iam-oidc-provider --cluster mlops-taxi --region us-east-1 --approve

3. Check for and delete stale IAM service account CloudFormation  
   stack first.
   aws cloudformation delete-stack --stack-name eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa --region us-east-1

4. Create service account:
   eksctl create iamserviceaccount \
    --cluster mlops-taxi \
    --region us-east-1 \
    --name mlops-taxi-sa \
    --namespace default \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
    --approve \
    --override-existing-serviceaccounts

5. Point kubectl at cluster:
   aws eks update-kubeconfig --region us-east-1 --name mlops-taxi

6. Deploy:
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml

7. Get load balancer URL (wait 2-3 min):
   kubectl get services

8. Update API_BASE in dashboard/src/PortfolioDashboard.jsx with new URL

9. Rebuild dashboard:
   cd dashboard && npm run build

## Tear Down (run after interview)

eksctl delete nodegroup --cluster mlops-taxi --region us-east-1 --name standard-workers
