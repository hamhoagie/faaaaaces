#!/bin/bash

# FAAAAACES Reverse Proxy Setup Script
# Automated setup for nginx or Apache reverse proxy

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN="your-domain.com"
EMAIL="admin@your-domain.com"
FAAAAACES_PORT="5005"

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${BLUE}🔧 $1${NC}"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        OS="unknown"
    fi
}

# Setup nginx
setup_nginx() {
    print_header "Setting up nginx reverse proxy"
    
    case $OS in
        "debian")
            sudo apt-get update
            sudo apt-get install -y nginx certbot python3-certbot-nginx
            ;;
        "redhat")
            sudo yum install -y nginx certbot python3-certbot-nginx
            ;;
        "macos")
            brew install nginx certbot
            ;;
        *)
            print_error "Unsupported OS for automatic nginx setup"
            return 1
            ;;
    esac
    
    # Create nginx config
    print_info "Creating nginx configuration..."
    
    # Get user input for domain
    read -p "Enter your domain name (e.g., example.com): " DOMAIN
    read -p "Enter your email for SSL certificates: " EMAIL
    
    # Copy and customize nginx config
    sudo cp "$PROJECT_ROOT/deploy/nginx.conf" "/etc/nginx/sites-available/faaaaaces"
    
    # Replace placeholders
    sudo sed -i "s/your-domain.com/$DOMAIN/g" "/etc/nginx/sites-available/faaaaaces"
    sudo sed -i "s|/path/to/faaaaaces|$PROJECT_ROOT|g" "/etc/nginx/sites-available/faaaaaces"
    
    # Enable site
    sudo ln -sf "/etc/nginx/sites-available/faaaaaces" "/etc/nginx/sites-enabled/"
    
    # Remove default site
    sudo rm -f "/etc/nginx/sites-enabled/default"
    
    # Test nginx config
    sudo nginx -t
    
    # Start/restart nginx
    if systemctl is-active --quiet nginx; then
        sudo systemctl reload nginx
    else
        sudo systemctl start nginx
        sudo systemctl enable nginx
    fi
    
    print_status "nginx configured successfully"
    
    # Setup SSL with Let's Encrypt
    if [ "$DOMAIN" != "your-domain.com" ] && [ "$DOMAIN" != "localhost" ]; then
        print_info "Setting up SSL certificates with Let's Encrypt..."
        sudo certbot --nginx -d "$DOMAIN" -d "www.$DOMAIN" --email "$EMAIL" --agree-tos --non-interactive
        print_status "SSL certificates configured"
    else
        print_warning "Skipping SSL setup - using placeholder domain"
    fi
}

# Setup Apache
setup_apache() {
    print_header "Setting up Apache reverse proxy"
    
    case $OS in
        "debian")
            sudo apt-get update
            sudo apt-get install -y apache2 certbot python3-certbot-apache
            
            # Enable required modules
            sudo a2enmod proxy
            sudo a2enmod proxy_http
            sudo a2enmod proxy_balancer
            sudo a2enmod lbmethod_byrequests
            sudo a2enmod ssl
            sudo a2enmod rewrite
            sudo a2enmod headers
            sudo a2enmod deflate
            sudo a2enmod expires
            sudo a2enmod remoteip
            ;;
        "redhat")
            sudo yum install -y httpd certbot python3-certbot-apache
            
            # Enable modules (different syntax for RHEL/CentOS)
            sudo sed -i '/LoadModule rewrite_module/s/^#//g' /etc/httpd/conf/httpd.conf
            sudo sed -i '/LoadModule ssl_module/s/^#//g' /etc/httpd/conf/httpd.conf
            ;;
        "macos")
            brew install httpd certbot
            ;;
        *)
            print_error "Unsupported OS for automatic Apache setup"
            return 1
            ;;
    esac
    
    # Get user input for domain
    read -p "Enter your domain name (e.g., example.com): " DOMAIN
    read -p "Enter your email for SSL certificates: " EMAIL
    
    # Copy and customize Apache config
    if [ "$OS" = "debian" ]; then
        APACHE_CONF_DIR="/etc/apache2/sites-available"
        APACHE_SITES_ENABLED="/etc/apache2/sites-enabled"
    else
        APACHE_CONF_DIR="/etc/httpd/conf.d"
        APACHE_SITES_ENABLED="/etc/httpd/conf.d"
    fi
    
    sudo cp "$PROJECT_ROOT/deploy/apache.conf" "$APACHE_CONF_DIR/faaaaaces.conf"
    
    # Replace placeholders
    sudo sed -i "s/your-domain.com/$DOMAIN/g" "$APACHE_CONF_DIR/faaaaaces.conf"
    sudo sed -i "s|/path/to/faaaaaces|$PROJECT_ROOT|g" "$APACHE_CONF_DIR/faaaaaces.conf"
    
    # Enable site (Debian/Ubuntu)
    if [ "$OS" = "debian" ]; then
        sudo a2ensite faaaaaces.conf
        sudo a2dissite 000-default.conf
    fi
    
    # Test Apache config
    if [ "$OS" = "debian" ]; then
        sudo apache2ctl configtest
    else
        sudo httpd -t
    fi
    
    # Start/restart Apache
    if [ "$OS" = "debian" ]; then
        if systemctl is-active --quiet apache2; then
            sudo systemctl reload apache2
        else
            sudo systemctl start apache2
            sudo systemctl enable apache2
        fi
    else
        if systemctl is-active --quiet httpd; then
            sudo systemctl reload httpd
        else
            sudo systemctl start httpd
            sudo systemctl enable httpd
        fi
    fi
    
    print_status "Apache configured successfully"
    
    # Setup SSL with Let's Encrypt
    if [ "$DOMAIN" != "your-domain.com" ] && [ "$DOMAIN" != "localhost" ]; then
        print_info "Setting up SSL certificates with Let's Encrypt..."
        if [ "$OS" = "debian" ]; then
            sudo certbot --apache -d "$DOMAIN" -d "www.$DOMAIN" --email "$EMAIL" --agree-tos --non-interactive
        else
            sudo certbot --apache -d "$DOMAIN" -d "www.$DOMAIN" --email "$EMAIL" --agree-tos --non-interactive
        fi
        print_status "SSL certificates configured"
    else
        print_warning "Skipping SSL setup - using placeholder domain"
    fi
}

# Setup Docker with nginx
setup_docker_nginx() {
    print_header "Setting up Docker with nginx reverse proxy"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose first."
        return 1
    fi
    
    # Get user input
    read -p "Enter your domain name (e.g., example.com): " DOMAIN
    read -p "Enter your email for SSL certificates: " EMAIL
    
    # Create docker-compose with nginx
    cat > "$PROJECT_ROOT/docker-compose.prod.yml" << EOF
version: '3.8'

services:
  faaaaaces:
    build: .
    container_name: faaaaaces-app
    expose:
      - "5005"
    volumes:
      - ./faaaaaces.db:/app/faaaaaces.db
      - ./faces:/app/faces
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./temp:/app/temp
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
    restart: unless-stopped
    networks:
      - faaaaaces-net

  nginx:
    image: nginx:alpine
    container_name: faaaaaces-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deploy/nginx-docker.conf:/etc/nginx/nginx.conf
      - ./deploy/ssl:/etc/ssl/certs
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - faaaaaces
    restart: unless-stopped
    networks:
      - faaaaaces-net

  certbot:
    image: certbot/certbot
    container_name: faaaaaces-certbot
    volumes:
      - ./deploy/ssl:/etc/letsencrypt
      - ./deploy/ssl-challenge:/var/www/certbot
    command: certonly --webroot --webroot-path=/var/www/certbot --email $EMAIL --agree-tos --no-eff-email -d $DOMAIN -d www.$DOMAIN

networks:
  faaaaaces-net:
    driver: bridge
EOF

    # Create simplified nginx config for Docker
    cat > "$PROJECT_ROOT/deploy/nginx-docker.conf" << EOF
events {
    worker_connections 1024;
}

http {
    upstream faaaaaces_backend {
        server faaaaaces:5005;
    }

    server {
        listen 80;
        server_name $DOMAIN www.$DOMAIN;
        
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }
        
        location / {
            return 301 https://\$server_name\$request_uri;
        }
    }

    server {
        listen 443 ssl;
        server_name $DOMAIN www.$DOMAIN;
        
        ssl_certificate /etc/ssl/certs/fullchain.pem;
        ssl_certificate_key /etc/ssl/certs/privkey.pem;
        
        client_max_body_size 1G;
        
        location / {
            proxy_pass http://faaaaaces_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

    print_status "Docker configuration created"
    print_info "To start: docker-compose -f docker-compose.prod.yml up -d"
}

# Main menu
show_menu() {
    echo ""
    print_header "FAAAAACES Reverse Proxy Setup"
    echo "Choose your preferred reverse proxy setup:"
    echo ""
    echo "1) nginx (recommended)"
    echo "2) Apache"
    echo "3) Docker + nginx"
    echo "4) Exit"
    echo ""
}

main() {
    detect_os
    print_info "Detected OS: $OS"
    
    while true; do
        show_menu
        read -p "Enter your choice (1-4): " choice
        
        case $choice in
            1)
                setup_nginx
                break
                ;;
            2)
                setup_apache
                break
                ;;
            3)
                setup_docker_nginx
                break
                ;;
            4)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please enter 1-4."
                ;;
        esac
    done
    
    echo ""
    print_status "Reverse proxy setup complete!"
    echo ""
    print_info "Next steps:"
    print_info "1. Start FAAAAACES: ./deploy/faaaaaces start"
    print_info "2. Check your domain: https://$DOMAIN"
    print_info "3. Monitor logs for any issues"
    echo ""
}

# Run main function
main "$@"