#!/bin/bash

# Hate Speech Detection API - Automated Setup Script
# This script automates the complete setup process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Main script
clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════╗
║  Hate Speech Detection API - Setup Script            ║
║  Automated installation and configuration             ║
╚═══════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Step 1: Check prerequisites
print_header "STEP 1: Checking Prerequisites"

MISSING_DEPS=0

if ! check_command python3; then
    MISSING_DEPS=1
fi

if ! check_command pip3; then
    MISSING_DEPS=1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if (( $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l) )); then
    print_success "Python version $PYTHON_VERSION (>= $REQUIRED_VERSION)"
else
    print_error "Python version $PYTHON_VERSION is too old (need >= $REQUIRED_VERSION)"
    MISSING_DEPS=1
fi

if [ $MISSING_DEPS -eq 1 ]; then
    print_error "Missing dependencies. Please install them first."
    exit 1
fi

# Step 2: Install Python dependencies
print_header "STEP 2: Installing Python Dependencies"

print_info "Installing core dependencies..."
pip3 install -r requirements.txt --quiet || {
    print_error "Failed to install core dependencies"
    exit 1
}
print_success "Core dependencies installed"

print_info "Installing API dependencies..."
pip3 install -r requirements_api.txt --quiet || {
    print_error "Failed to install API dependencies"
    exit 1
}
print_success "API dependencies installed"

# Step 3: Download NLTK data
print_header "STEP 3: Downloading NLTK Data"

python3 << EOF
import nltk
import sys

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "NLTK data downloaded"
else
    print_warning "NLTK data download had issues (non-critical)"
fi

# Step 4: Check for trained models
print_header "STEP 4: Checking for Trained Models"

if [ -d "saved_models" ] && [ "$(ls -A saved_models)" ]; then
    print_success "Trained models found"
    MODEL_EXISTS=1
else
    print_warning "No trained models found"
    print_info "Models need to be trained before starting the API"
    MODEL_EXISTS=0
    
    read -p "Do you want to train models now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_header "Training Models (This will take 10-30 minutes)"
        
        # Check if data exists
        if [ ! -f "data/raw/labeled_data.csv" ]; then
            print_error "Training data not found at data/raw/labeled_data.csv"
            print_info "Please place your dataset in the data/raw/ directory"
            exit 1
        fi
        
        print_info "Starting model training..."
        python3 main_train.py || {
            print_error "Model training failed"
            exit 1
        }
        print_success "Models trained successfully"
        MODEL_EXISTS=1
    fi
fi

# Step 5: Create necessary directories
print_header "STEP 5: Creating Directories"

mkdir -p logs
mkdir -p saved_models
mkdir -p results
print_success "Directories created"

# Step 6: Test API
print_header "STEP 6: Testing API"

if [ $MODEL_EXISTS -eq 1 ]; then
    print_info "Starting API test..."
    
    # Start API in background
    python3 main.py &
    API_PID=$!
    
    # Wait for API to start
    sleep 5
    
    # Test health endpoint
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "API is responding"
        
        # Kill test API
        kill $API_PID 2>/dev/null
        wait $API_PID 2>/dev/null
        
        print_success "API test completed"
    else
        print_warning "API health check failed (might be fine)"
        kill $API_PID 2>/dev/null
        wait $API_PID 2>/dev/null
    fi
else
    print_warning "Skipping API test (no trained models)"
fi

# Step 7: Create start script
print_header "STEP 7: Creating Start Scripts"

# Create start_api.sh
cat > start_api.sh << 'EOF'
#!/bin/bash
echo "Starting Hate Speech Detection API..."
python3 main.py
EOF
chmod +x start_api.sh
print_success "Created start_api.sh"

# Create start_api_production.sh
cat > start_api_production.sh << 'EOF'
#!/bin/bash
echo "Starting Hate Speech Detection API (Production Mode)..."
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --log-level info
EOF
chmod +x start_api_production.sh
print_success "Created start_api_production.sh"

# Step 8: Summary
print_header "SETUP COMPLETE! 🎉"

echo -e "${GREEN}"
cat << "EOF"
┌─────────────────────────────────────────────────────┐
│  Setup completed successfully!                      │
└─────────────────────────────────────────────────────┘
EOF
echo -e "${NC}"

echo -e "\n${BLUE}Next Steps:${NC}\n"

if [ $MODEL_EXISTS -eq 1 ]; then
    echo -e "1. ${GREEN}Start the API:${NC}"
    echo -e "   ${YELLOW}./start_api.sh${NC}"
    echo -e "   or"
    echo -e "   ${YELLOW}python3 main.py${NC}\n"
    
    echo -e "2. ${GREEN}Access the API:${NC}"
    echo -e "   • Swagger UI:  ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "   • ReDoc:       ${YELLOW}http://localhost:8000/redoc${NC}"
    echo -e "   • Health:      ${YELLOW}http://localhost:8000/health${NC}\n"
    
    echo -e "3. ${GREEN}Test the API:${NC}"
    echo -e "   ${YELLOW}python3 test_api.py${NC}\n"
    
    echo -e "4. ${GREEN}Try the web demo:${NC}"
    echo -e "   Open ${YELLOW}demo.html${NC} in your browser\n"
else
    echo -e "1. ${GREEN}Train models first:${NC}"
    echo -e "   ${YELLOW}python3 main_train.py${NC}\n"
    
    echo -e "2. ${GREEN}Then start the API:${NC}"
    echo -e "   ${YELLOW}./start_api.sh${NC}\n"
fi

echo -e "${BLUE}Production Deployment:${NC}"
echo -e "   • See ${YELLOW}DEPLOYMENT.md${NC} for production setup"
echo -e "   • Use ${YELLOW}./start_api_production.sh${NC} for production mode\n"

echo -e "${BLUE}Documentation:${NC}"
echo -e "   • API Guide:        ${YELLOW}README_API.md${NC}"
echo -e "   • Deployment:       ${YELLOW}DEPLOYMENT.md${NC}"
echo -e "   • Main Project:     ${YELLOW}README.md${NC}\n"

echo -e "${GREEN}Setup completed successfully! 🚀${NC}\n"

# Optional: Ask to start API now
if [ $MODEL_EXISTS -eq 1 ]; then
    read -p "Do you want to start the API now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Starting API..."
        ./start_api.sh
    fi
fi