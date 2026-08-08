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

**Verify it actually landed before moving on** — `eksctl` can silently no-op if it thinks the
service account already exists (see Known Issues below):

```bash
kubectl get serviceaccount -n default
```

You should see `mlops-taxi-sa` listed alongside `default`. If it's missing, stop and see
**Known Issue #1** below before proceeding — deployments will fail later with a
`serviceaccount not found` error if this isn't fixed now.

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

### 1. Delete the IAM service account stack explicitly (before deleting the cluster)

`eksctl delete cluster` does NOT reliably remove the `iamserviceaccount` CloudFormation
stack if termination protection is enabled on it — this has caused a stale-stack issue on
re-spin-up before (see Known Issue #1). Delete it explicitly first:

```bash
aws cloudformation update-termination-protection \
  --stack-name eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa \
  --no-enable-termination-protection \
  --region us-east-1

aws cloudformation delete-stack \
  --stack-name eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa \
  --region us-east-1
```

If the stack doesn't exist (e.g. first-ever teardown), the first command will just error
harmlessly — safe to ignore and move on.

### 2. Delete entire cluster (~15-20 min)

```bash
eksctl delete cluster --name mlops-taxi --region us-east-1
```

### This removes the remaining resources: nodes, control plane, load balancer, and OIDC provider.

### 3. Verify clean teardown

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

## Known Issues

### 1. `mlops-taxi-sa` service account silently not created ("no tasks")

**Symptom:** Step 3's `eksctl create iamserviceaccount` command runs without error but prints
`no tasks`, and `kubectl get serviceaccount -n default` shows only `default` — `mlops-taxi-sa`
is missing. Deployments created later fail with:

**Cause:** `eksctl` tracks IAM service accounts via a CloudFormation stack
(`eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa`). If a previous cluster
session's stack wasn't cleaned up (e.g. teardown didn't remove it, or it has termination
protection enabled), `eksctl` sees the stack still exists and assumes the service account is
already set up — even though the _cluster_ itself was deleted and rebuilt fresh, so the
service account doesn't actually exist in the new cluster's Kubernetes API.

**Fix:**

```bash
aws cloudformation update-termination-protection \
  --stack-name eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa \
  --no-enable-termination-protection \
  --region us-east-1

aws cloudformation delete-stack \
  --stack-name eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa \
  --region us-east-1

# confirm it's actually gone before retrying — should return a "does not exist" error
aws cloudformation describe-stacks \
  --stack-name eksctl-mlops-taxi-addon-iamserviceaccount-default-mlops-taxi-sa \
  --region us-east-1
```

Then re-run Step 3's `eksctl create iamserviceaccount` command — it should now actually create
resources instead of printing `no tasks`. Verify with `kubectl get serviceaccount -n default`.

**Permanent fix:** Teardown Step 1 (above) now deletes this stack explicitly before deleting
the cluster, which should prevent this from recurring on future spin-ups.

---

## Cost Reference

| State               | Resources                        | Est. Monthly Cost |
| ------------------- | -------------------------------- | ----------------- |
| Fully torn down     | ECR + S3 only                    | ~$1–2             |
| Spun up (interview) | 2x t3.small + control plane + LB | ~$118             |
