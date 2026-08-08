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
service account already exists (see Known Issue #1):

```bash
kubectl get serviceaccount -n default
```

You should see `mlops-taxi-sa` listed alongside `default`. If it's missing, stop and see
**Known Issue #1** below before proceeding.

### 4. Update kubeconfig

```bash
aws eks update-kubeconfig --region us-east-1 --name mlops-taxi
```

**Verify `kubectl` actually switched to this cluster before proceeding** — it's easy for
`kubectl` to still be pointed at `docker-desktop` or a stale context (see Known Issue #2):

```bash
kubectl config current-context
```

This must show `arn:aws:eks:us-east-1:665012226357:cluster/mlops-taxi` — NOT
`docker-desktop` or anything else. If it shows the wrong context, re-run the
`update-kubeconfig` command above and check again before moving on. Every `kubectl` command
below silently talks to whatever context this shows, with no warning if it's the wrong one.

### 5. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 6. Get new load balancer URL (wait 2-3 min for provisioning)

```bash
kubectl get services
```

Copy the `EXTERNAL-IP` value from `mlops-taxi-service`. **Note: the service listens on port
80, not 8080** — check the `PORT(S)` column (e.g. `80:31079/TCP`); the external port is the
first number.

**Verify the deploy actually works, including caching**, before moving on to Vercel:

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tpep_pickup_datetime": "2026-08-07T14:30:00",
    "Pickup_longitude": -73.985,
    "Pickup_latitude": 40.758,
    "Dropoff_longitude": -73.968,
    "Dropoff_latitude": 40.785,
    "Passenger_count": 1,
    "Trip_distance": 3.2
  }'
```

Run it twice with the identical payload. First call should return `"source":"model"` (cache
miss, ran real inference). Second call should return `"source":"cache"` (served from
ElastiCache, no inference). If both calls say `"source":"model"`, Redis isn't being reached —
check `REDIS_HOST` in `k8s/deployment.yaml` matches the current `terraform output
redis_endpoint`, since this changes every time the ElastiCache cluster is recreated.

**Check the latency difference in Prometheus/Grafana** (already instrumented via
`Instrumentator()` in `main.py`) — cache hits should show noticeably lower request latency
than cache misses. This is the actual evidence for "caching reduced latency," not just proof
the code runs.

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

### 3. Tear down the ElastiCache/Redis infrastructure (managed separately, via Terraform)

```bash
cd terraform
terraform destroy
```

Confirm with `yes` when prompted. This removes the ElastiCache cluster, subnet group, and
security group. Skipping this step leaves Redis running and billing continuously, since it's
outside the `eksctl delete cluster` scope entirely.

### 4. Verify clean teardown

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

# No ElastiCache clusters
aws elasticache describe-cache-clusters --region us-east-1 \
  --query 'CacheClusters[*].CacheClusterId' \
  --output table
```

All four should return empty. After teardown you are only paying for ECR and S3 (effectively free).

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

Then re-run Step 3's `eksctl create iamserviceaccount` command. Verify with
`kubectl get serviceaccount -n default`.

**Permanent fix:** Teardown Step 1 (above) now deletes this stack explicitly before deleting
the cluster, which should prevent this from recurring on future spin-ups.

### 2. `kubectl` silently pointed at the wrong cluster (`docker-desktop`)

**Symptom:** `kubectl` commands run without error, but nothing behaves as expected —
`kubectl get pods` shows "No resources found" even after applying a Deployment, or
`kubectl get serviceaccount` doesn't show an account you just confirmed exists via `eksctl`.
No error message points to the actual cause.

**Cause:** `kubectl` operates against whatever context is currently active
(`kubectl config current-context`), and this can silently be `docker-desktop` (Docker
Desktop's local Kubernetes) instead of the EKS cluster — especially if Step 4's
`update-kubeconfig` command was run but never verified. `eksctl` commands are unaffected by
this — they always talk to the named EKS cluster directly — so `eksctl` can report success
while `kubectl` is checking a completely different, unrelated cluster (Docker Desktop's,
which is typically empty).

**Fix:**

```bash
kubectl config current-context
```

If this doesn't show `arn:aws:eks:us-east-1:665012226357:cluster/mlops-taxi`, run:

```bash
aws eks update-kubeconfig --region us-east-1 --name mlops-taxi
```

Then re-check `kubectl config current-context` again before continuing with any `kubectl`
commands.

**Permanent fix:** Step 4 above now includes this verification check inline, so this should
be caught immediately during spin-up rather than discovered later while debugging an
unrelated-seeming failure.

### 3. Pods `CrashLoopBackOff` with `exec format error` in logs

**Symptom:** Pods schedule and start (unlike Known Issue #1/#2, which prevent pods from being
created at all), but immediately crash. `kubectl logs <pod-name>` shows:
