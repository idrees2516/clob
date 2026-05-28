#!/bin/bash
set -e

# Advanced Trading System Deployment Script
# This script handles the complete deployment of the trading system
# with performance optimizations and monitoring setup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
NAMESPACE="${NAMESPACE:-trading-system}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed (optional)
    if command -v helm &> /dev/null; then
        log_info "Helm detected, will use for advanced deployments"
        HELM_AVAILABLE=true
    else
        log_warning "Helm not found, using kubectl for deployment"
        HELM_AVAILABLE=false
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if we have sufficient permissions
    if ! kubectl auth can-i create namespace &> /dev/null; then
        log_error "Insufficient permissions to create namespace"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build with performance optimizations
    docker build \
        --file deployment/Dockerfile \
        --tag "advanced-trading-system:${DEPLOYMENT_ENV}" \
        --tag "advanced-trading-system:latest" \
        --build-arg RUST_FLAGS="-C target-cpu=native -C target-feature=+avx2,+avx512f" \
        --build-arg BUILD_MODE=release \
        .
    
    log_success "Docker image built successfully"
}

# Push image to registry (if configured)
push_image() {
    if [ -n "$DOCKER_REGISTRY" ]; then
        log_info "Pushing image to registry: $DOCKER_REGISTRY"
        
        docker tag "advanced-trading-system:${DEPLOYMENT_ENV}" \
                   "$DOCKER_REGISTRY/advanced-trading-system:${DEPLOYMENT_ENV}"
        
        docker push "$DOCKER_REGISTRY/advanced-trading-system:${DEPLOYMENT_ENV}"
        
        log_success "Image pushed to registry"
    else
        log_info "No registry configured, skipping image push"
    fi
}

# Setup system prerequisites on nodes
setup_node_prerequisites() {
    log_info "Setting up node prerequisites..."
    
    # Create DaemonSet for node configuration
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-setup
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: node-setup
  template:
    metadata:
      labels:
        name: node-setup
    spec:
      hostPID: true
      hostNetwork: true
      containers:
      - name: node-setup
        image: busybox:1.35
        command:
        - /bin/sh
        - -c
        - |
          # Configure huge pages
          echo 2048 > /host/proc/sys/vm/nr_hugepages
          
          # Configure CPU governor
          for cpu in /host/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -w "\$cpu" ]; then
              echo performance > "\$cpu"
            fi
          done
          
          # Configure network settings
          echo 50 > /host/proc/sys/net/core/busy_read
          echo 50 > /host/proc/sys/net/core/busy_poll
          
          # Configure kernel parameters for low latency
          echo 1000000 > /host/proc/sys/kernel/sched_rt_period_us
          echo -1 > /host/proc/sys/kernel/sched_rt_runtime_us
          
          # Keep container running
          sleep infinity
        securityContext:
          privileged: true
        volumeMounts:
        - name: host-proc
          mountPath: /host/proc
        - name: host-sys
          mountPath: /host/sys
      volumes:
      - name: host-proc
        hostPath:
          path: /proc
      - name: host-sys
        hostPath:
          path: /sys
      tolerations:
      - operator: Exists
EOF
    
    log_success "Node prerequisites configured"
}

# Create namespace and RBAC
setup_namespace() {
    log_info "Setting up namespace and RBAC..."
    
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/namespace.yaml"
    
    log_success "Namespace and RBAC configured"
}

# Deploy storage components
deploy_storage() {
    log_info "Deploying storage components..."
    
    # Create storage class and persistent volumes
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/storage.yaml"
    
    # Wait for PVs to be available
    kubectl wait --for=condition=Available pv/trading-data-pv --timeout=60s
    kubectl wait --for=condition=Available pv/trading-logs-pv --timeout=60s
    kubectl wait --for=condition=Available pv/trading-backup-pv --timeout=60s
    
    log_success "Storage components deployed"
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    
    # Apply ConfigMaps
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/configmap.yaml"
    
    # Create secrets (if they don't exist)
    if ! kubectl get secret trading-system-secrets -n "$NAMESPACE" &> /dev/null; then
        kubectl create secret generic trading-system-secrets \
            --namespace="$NAMESPACE" \
            --from-literal=database-password="$(openssl rand -base64 32)" \
            --from-literal=redis-password="$(openssl rand -base64 32)" \
            --from-literal=api-key="$(openssl rand -base64 32)"
    fi
    
    log_success "Configuration deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    kubectl create configmap prometheus-config \
        --namespace=monitoring \
        --from-file="$PROJECT_ROOT/deployment/monitoring/prometheus.yml" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create configmap prometheus-rules \
        --namespace=monitoring \
        --from-file="$PROJECT_ROOT/deployment/monitoring/alert-rules.yml" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus using kubectl
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: rules
          mountPath: /etc/prometheus/rules
        - name: data
          mountPath: /prometheus
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time=7d'
        - '--web.enable-lifecycle'
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: rules
        configMap:
          name: prometheus-rules
      - name: data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
EOF
    
    log_success "Monitoring stack deployed"
}

# Deploy main application
deploy_application() {
    log_info "Deploying trading system application..."
    
    # Update image in deployment if registry is configured
    if [ -n "$DOCKER_REGISTRY" ]; then
        sed -i.bak "s|advanced-trading-system:latest|$DOCKER_REGISTRY/advanced-trading-system:${DEPLOYMENT_ENV}|g" \
            "$PROJECT_ROOT/deployment/kubernetes/deployment.yaml"
    fi
    
    # Apply deployment
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/deployment.yaml"
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available deployment/trading-system \
        --namespace="$NAMESPACE" --timeout=300s
    
    # Restore original deployment file if modified
    if [ -f "$PROJECT_ROOT/deployment/kubernetes/deployment.yaml.bak" ]; then
        mv "$PROJECT_ROOT/deployment/kubernetes/deployment.yaml.bak" \
           "$PROJECT_ROOT/deployment/kubernetes/deployment.yaml"
    fi
    
    log_success "Trading system application deployed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -l app=trading-system
    
    # Check if pods are ready
    if ! kubectl wait --for=condition=ready pod -l app=trading-system \
         --namespace="$NAMESPACE" --timeout=60s; then
        log_error "Pods are not ready"
        kubectl describe pods -n "$NAMESPACE" -l app=trading-system
        exit 1
    fi
    
    # Check service endpoints
    kubectl get endpoints -n "$NAMESPACE"
    
    # Test health endpoint
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app=trading-system -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "$NAMESPACE" "$POD_NAME" -- curl -f http://localhost:8081/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        kubectl logs -n "$NAMESPACE" "$POD_NAME" --tail=50
        exit 1
    fi
    
    # Test metrics endpoint
    if kubectl exec -n "$NAMESPACE" "$POD_NAME" -- curl -f http://localhost:8080/metrics > /dev/null; then
        log_success "Metrics endpoint accessible"
    else
        log_warning "Metrics endpoint not accessible"
    fi
    
    log_success "Deployment verification completed"
}

# Performance tuning
apply_performance_tuning() {
    log_info "Applying performance tuning..."
    
    # Create performance tuning job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: performance-tuning
  namespace: $NAMESPACE
spec:
  template:
    spec:
      hostPID: true
      hostNetwork: true
      containers:
      - name: tuning
        image: busybox:1.35
        command:
        - /bin/sh
        - -c
        - |
          # CPU performance tuning
          for cpu in /host/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -w "\$cpu" ]; then
              echo performance > "\$cpu"
            fi
          done
          
          # Network performance tuning
          echo 1 > /host/proc/sys/net/core/busy_read
          echo 1 > /host/proc/sys/net/core/busy_poll
          
          # Memory performance tuning
          echo 1 > /host/proc/sys/vm/swappiness
          
          echo "Performance tuning completed"
        securityContext:
          privileged: true
        volumeMounts:
        - name: host-proc
          mountPath: /host/proc
        - name: host-sys
          mountPath: /host/sys
      volumes:
      - name: host-proc
        hostPath:
          path: /proc
      - name: host-sys
        hostPath:
          path: /sys
      restartPolicy: OnFailure
EOF
    
    # Wait for job completion
    kubectl wait --for=condition=complete job/performance-tuning \
        --namespace="$NAMESPACE" --timeout=60s
    
    log_success "Performance tuning applied"
}

# Setup monitoring and alerting
setup_alerting() {
    log_info "Setting up alerting..."
    
    # Create AlertManager configuration
    kubectl create configmap alertmanager-config \
        --namespace=monitoring \
        --from-literal=alertmanager.yml="
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@trading-system.com'

route:
  group_by: ['alertname']
  group_wait: 1s
  group_interval: 1s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'ops@trading-system.com'
    subject: 'Trading System Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
" --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Alerting configured"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary resources..."
    
    # Remove temporary jobs
    kubectl delete job performance-tuning -n "$NAMESPACE" --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting deployment of Advanced Trading System"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Namespace: $NAMESPACE"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    build_image
    push_image
    setup_node_prerequisites
    setup_namespace
    deploy_storage
    deploy_config
    deploy_monitoring
    deploy_application
    verify_deployment
    apply_performance_tuning
    setup_alerting
    
    log_success "Deployment completed successfully!"
    
    # Display access information
    echo ""
    log_info "Access Information:"
    echo "  Metrics: kubectl port-forward -n $NAMESPACE svc/trading-system-service 8080:8080"
    echo "  Health:  kubectl port-forward -n $NAMESPACE svc/trading-system-service 8081:8081"
    echo "  Admin:   kubectl port-forward -n $NAMESPACE svc/trading-system-service 8082:8082"
    echo "  Prometheus: kubectl port-forward -n monitoring svc/prometheus 9090:9090"
    echo ""
    log_info "To check system status:"
    echo "  kubectl get pods -n $NAMESPACE"
    echo "  kubectl logs -n $NAMESPACE -l app=trading-system -f"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env ENV          Deployment environment (default: production)"
            echo "  --namespace NS     Kubernetes namespace (default: trading-system)"
            echo "  --registry REG     Docker registry URL"
            echo "  --skip-build       Skip Docker image build"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main