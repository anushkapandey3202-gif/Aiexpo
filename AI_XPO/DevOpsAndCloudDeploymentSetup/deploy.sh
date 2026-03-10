#!/usr/bin/env bash
# =============================================================================
# SentinelAI — Production Deployment Script
# Builds, tags, pushes Docker images to ECR, then applies K8s manifests
# Usage: ./scripts/deploy.sh [--env prod|staging] [--service api|ml|router|all]
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Configuration ─────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TIMESTAMP=$(date +%Y%m%d%H%M%S)
readonly GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)
readonly IMAGE_TAG="${GIT_SHA}-${TIMESTAMP}"

# Defaults (override via environment or flags)
ENV="${DEPLOY_ENV:-prod}"
SERVICE="${DEPLOY_SERVICE:-all}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID must be set}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
K8S_NAMESPACE="sentinelai"
K8S_CONTEXT="${K8S_CONTEXT:-arn:aws:eks:${AWS_REGION}:${AWS_ACCOUNT_ID}:cluster/sentinelai-${ENV}}"

# Color output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)     ENV="$2";     shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --tag)     IMAGE_TAG="$2"; shift 2 ;;
    *) log_error "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Prerequisite checks ───────────────────────────────────────────────────────
check_prerequisites() {
  local tools=(docker kubectl aws git)
  for tool in "${tools[@]}"; do
    command -v "$tool" &>/dev/null || { log_error "$tool not found in PATH"; exit 1; }
  done
  log_info "Prerequisites OK"
}

# ── ECR Authentication ────────────────────────────────────────────────────────
ecr_login() {
  log_info "Authenticating with ECR (${AWS_REGION})..."
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY"
}

# ── Build + Push image ────────────────────────────────────────────────────────
build_and_push() {
  local service="$1"
  local dockerfile="$2"
  local ecr_repo="${ECR_REGISTRY}/sentinelai/${service}"

  log_info "Building ${service} → ${ecr_repo}:${IMAGE_TAG}"

  # Create ECR repo if absent
  aws ecr describe-repositories --repository-names "sentinelai/${service}" \
    --region "$AWS_REGION" &>/dev/null \
    || aws ecr create-repository --repository-name "sentinelai/${service}" \
       --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true \
       --encryption-configuration encryptionType=AES256

  # Build
  docker build \
    --file "$dockerfile" \
    --tag "${ecr_repo}:${IMAGE_TAG}" \
    --tag "${ecr_repo}:latest" \
    --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --build-arg GIT_SHA="$GIT_SHA" \
    --cache-from "${ecr_repo}:latest" \
    --target production \
    "$PROJECT_ROOT"

  # Push both tags
  docker push "${ecr_repo}:${IMAGE_TAG}"
  docker push "${ecr_repo}:latest"

  log_info "Pushed ${service}:${IMAGE_TAG} ✓"
}

# ── K8s deployment ────────────────────────────────────────────────────────────
deploy_k8s() {
  local service="$1"
  local deployment_name="sentinelai-${service}"

  log_info "Deploying ${deployment_name} to K8s (${ENV})..."

  kubectl config use-context "$K8S_CONTEXT"

  # Apply all base manifests
  kubectl apply -f "$PROJECT_ROOT/k8s/base/" --namespace "$K8S_NAMESPACE"
  kubectl apply -f "$PROJECT_ROOT/k8s/autoscaling/" --namespace "$K8S_NAMESPACE"
  kubectl apply -f "$PROJECT_ROOT/k8s/monitoring/" --namespace "$K8S_NAMESPACE"

  # Patch the deployment image tag
  kubectl set image \
    "deployment/${deployment_name}" \
    "${service}=${ECR_REGISTRY}/sentinelai/${service}:${IMAGE_TAG}" \
    --namespace "$K8S_NAMESPACE"

  # Annotate with deployment metadata
  kubectl annotate deployment "${deployment_name}" \
    "deployment.sentinelai.io/deployed-by=$(git config user.email || echo CI)" \
    "deployment.sentinelai.io/git-sha=${GIT_SHA}" \
    "deployment.sentinelai.io/timestamp=${TIMESTAMP}" \
    --overwrite \
    --namespace "$K8S_NAMESPACE"

  # Wait for rollout
  log_info "Waiting for rollout: ${deployment_name}..."
  if ! kubectl rollout status "deployment/${deployment_name}" \
       --namespace "$K8S_NAMESPACE" \
       --timeout=300s; then
    log_error "Rollout failed — initiating rollback"
    kubectl rollout undo "deployment/${deployment_name}" --namespace "$K8S_NAMESPACE"
    exit 1
  fi

  log_info "Rollout complete: ${deployment_name} ✓"
}

# ── Smoke tests ───────────────────────────────────────────────────────────────
smoke_test() {
  local api_url="https://api.sentinelai.yourorg.com"
  log_info "Running smoke tests against ${api_url}..."

  for endpoint in /health/live /health/ready; do
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "${api_url}${endpoint}" || echo "000")
    if [[ "$status" != "200" ]]; then
      log_error "Smoke test FAILED: ${endpoint} returned HTTP ${status}"
      exit 1
    fi
    log_info "  ${endpoint} → HTTP 200 ✓"
  done
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  log_info "SentinelAI Deploy | env=${ENV} service=${SERVICE} tag=${IMAGE_TAG}"

  check_prerequisites
  ecr_login

  case "$SERVICE" in
    api)
      build_and_push "api"    "$PROJECT_ROOT/dockerfiles/Dockerfile.api"
      deploy_k8s "api"
      ;;
    ml)
      build_and_push "ml"     "$PROJECT_ROOT/dockerfiles/Dockerfile.ml"
      deploy_k8s "ml"
      ;;
    router)
      build_and_push "router" "$PROJECT_ROOT/dockerfiles/Dockerfile.router"
      deploy_k8s "router"
      ;;
    all)
      build_and_push "api"    "$PROJECT_ROOT/dockerfiles/Dockerfile.api"
      build_and_push "ml"     "$PROJECT_ROOT/dockerfiles/Dockerfile.ml"
      build_and_push "router" "$PROJECT_ROOT/dockerfiles/Dockerfile.router"
      deploy_k8s "api"
      deploy_k8s "ml"
      deploy_k8s "router"
      ;;
    *)
      log_error "Unknown service: ${SERVICE}. Use api|ml|router|all"
      exit 1
      ;;
  esac

  smoke_test
  log_info "Deployment complete ✓  tag=${IMAGE_TAG}"
}

main "$@"
