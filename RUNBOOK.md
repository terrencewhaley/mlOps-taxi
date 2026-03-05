# MLOps Taxi — Interview Spin-Up Runbook

## Spin Up (run before interview)

### 1. Create cluster + node group (~15-20 min)

```bash
eksctl create cluster \
  --name mlops-taxi \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.small \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed
```

### 2. Associate OIDC provider

```bash
eksctl utils associate-iam-oidc-provider \
  --cluster mlops-taxi \
  --region us-east-1 \
  --approve
```

### 3. Create IAM service account (S3 access)

```bash
eksctl create iamserviceaccount \
  --cluster mlops-taxi \
  --region us-east-1 \
  --name mlops-taxi-sa \
  --namespace default \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
  --approve \
  --override-existing-serviceaccounts
```

### 4. Update kubeconfig

```bash
aws eks update-kubeconfig --region us-east-1 --name mlops-taxi
```

### 5. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 6. Get new load balancer URL (wait 2-3 min for provisioning)

```bash
kubectl get services
```

Copy the `EXTERNAL-IP` value from `mlops-taxi-service`.

### 7. Update Vercel with new load balancer URL

Edit `vercel.json` — replace the old load balancer URL with the new one, then:

```bash
git add vercel.json && git commit -m "Update load balancer URL" && git push
```

Vercel will auto-redeploy in ~1 min. Verify the dashboard API status shows green.

---

## Teardown (run after interview)

### 1. Delete entire cluster (~15-20 min)

```bash
eksctl delete cluster --name mlops-taxi --region us-east-1
```

### This removes everything: nodes, control plane, load balancer, IAM service account, OIDC provider, and all CloudFormation stacks.

### 2. Verify clean teardown

```bash
# Cluster is gone
aws eks list-clusters --region us-east-1

# No running EC2 instances
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].{ID:InstanceId,Type:InstanceType}' \
  --output table

# No load balancers
aws elbv2 describe-load-balancers --region us-east-1 \
  --query 'LoadBalancers[*].LoadBalancerName' \
  --output table
```

All three should return empty. After teardown you are only paying for ECR and S3 (effectively free).

---

## Cost Reference

| State               | Resources                        | Est. Monthly Cost |
| ------------------- | -------------------------------- | ----------------- |
| Fully torn down     | ECR + S3 only                    | ~$1–2             |
| Spun up (interview) | 2x t3.small + control plane + LB | ~$118             |
