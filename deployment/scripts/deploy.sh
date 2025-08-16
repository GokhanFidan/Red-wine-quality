#!/bin/bash

# Wine Quality Prediction API - Deployment Script
# Production deployment script for Docker containers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="wine-quality-prediction"
API_SERVICE="wine-api"
NGINX_SERVICE="nginx"
COMPOSE_FILE="docker-compose.yml"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking deployment requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log_info "All requirements satisfied."
}

build_images() {
    log_info "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --no-cache
    log_info "Docker images built successfully."
}

deploy_basic() {
    log_info "Deploying basic API service..."
    docker-compose -f $COMPOSE_FILE up -d $API_SERVICE
    log_info "Basic deployment completed."
}

deploy_production() {
    log_info "Deploying production services with Nginx..."
    docker-compose -f $COMPOSE_FILE --profile production up -d
    log_info "Production deployment completed."
}

deploy_full() {
    log_info "Deploying all services (API, Nginx, PostgreSQL, Redis)..."
    docker-compose -f $COMPOSE_FILE --profile production --profile analytics --profile caching up -d
    log_info "Full deployment completed."
}

health_check() {
    log_info "Performing health checks..."
    
    # Wait for services to start
    sleep 10
    
    # Check API health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_info "API service is healthy."
    else
        log_warning "API service health check failed."
    fi
    
    # Check Nginx (if deployed)
    if docker-compose -f $COMPOSE_FILE ps | grep -q nginx; then
        if curl -f http://localhost/health &> /dev/null; then
            log_info "Nginx service is healthy."
        else
            log_warning "Nginx service health check failed."
        fi
    fi
}

show_status() {
    log_info "Deployment status:"
    docker-compose -f $COMPOSE_FILE ps
    
    log_info "Service URLs:"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  API Health Check: http://localhost:8000/health"
    
    if docker-compose -f $COMPOSE_FILE ps | grep -q nginx; then
        echo "  Nginx Proxy: http://localhost"
    fi
}

stop_services() {
    log_info "Stopping all services..."
    docker-compose -f $COMPOSE_FILE down
    log_info "All services stopped."
}

cleanup() {
    log_info "Cleaning up Docker resources..."
    docker-compose -f $COMPOSE_FILE down -v --remove-orphans
    docker system prune -f
    log_info "Cleanup completed."
}

show_logs() {
    local service=${1:-$API_SERVICE}
    log_info "Showing logs for service: $service"
    docker-compose -f $COMPOSE_FILE logs -f $service
}

backup_data() {
    log_info "Backing up persistent data..."
    
    # Create backup directory
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $BACKUP_DIR
    
    # Backup volumes
    docker run --rm -v wine_models:/data -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/models.tar.gz -C /data .
    docker run --rm -v wine_plots:/data -v $(pwd)/$BACKUP_DIR:/backup alpine tar czf /backup/plots.tar.gz -C /data .
    
    log_info "Backup created in: $BACKUP_DIR"
}

restore_data() {
    local backup_dir=$1
    
    if [ -z "$backup_dir" ]; then
        log_error "Please specify backup directory."
        exit 1
    fi
    
    if [ ! -d "$backup_dir" ]; then
        log_error "Backup directory does not exist: $backup_dir"
        exit 1
    fi
    
    log_info "Restoring data from: $backup_dir"
    
    # Restore volumes
    if [ -f "$backup_dir/models.tar.gz" ]; then
        docker run --rm -v wine_models:/data -v $(pwd)/$backup_dir:/backup alpine tar xzf /backup/models.tar.gz -C /data
        log_info "Models restored."
    fi
    
    if [ -f "$backup_dir/plots.tar.gz" ]; then
        docker run --rm -v wine_plots:/data -v $(pwd)/$backup_dir:/backup alpine tar xzf /backup/plots.tar.gz -C /data
        log_info "Plots restored."
    fi
}

show_help() {
    echo "Wine Quality Prediction API - Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy          Deploy basic API service"
    echo "  deploy-prod     Deploy production services (API + Nginx)"
    echo "  deploy-full     Deploy all services (API + Nginx + PostgreSQL + Redis)"
    echo "  build           Build Docker images"
    echo "  status          Show service status"
    echo "  health          Perform health checks"
    echo "  logs [service]  Show logs for service (default: wine-api)"
    echo "  stop            Stop all services"
    echo "  cleanup         Stop services and clean up resources"
    echo "  backup          Backup persistent data"
    echo "  restore <dir>   Restore data from backup directory"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                    # Deploy basic API"
    echo "  $0 deploy-prod              # Deploy with Nginx"
    echo "  $0 logs wine-api            # Show API logs"
    echo "  $0 restore backups/20240116 # Restore from backup"
}

# Main script logic
case "$1" in
    "deploy")
        check_requirements
        build_images
        deploy_basic
        health_check
        show_status
        ;;
    "deploy-prod")
        check_requirements
        build_images
        deploy_production
        health_check
        show_status
        ;;
    "deploy-full")
        check_requirements
        build_images
        deploy_full
        health_check
        show_status
        ;;
    "build")
        check_requirements
        build_images
        ;;
    "status")
        show_status
        ;;
    "health")
        health_check
        ;;
    "logs")
        show_logs $2
        ;;
    "stop")
        stop_services
        ;;
    "cleanup")
        cleanup
        ;;
    "backup")
        backup_data
        ;;
    "restore")
        restore_data $2
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac