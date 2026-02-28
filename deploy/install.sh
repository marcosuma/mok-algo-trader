#!/bin/bash
#
# Server Installation Script for Trading Bot
# Run this on the GCP instance after cloning the repository
#

set -e

# Configuration
APP_DIR="${APP_DIR:-/opt/trading-bot}"
PYTHON_VERSION="python3.12"
LOG_DIR="$APP_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Trading Bot - Server Installation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root or with sudo capability
if [ "$EUID" -ne 0 ] && ! sudo -v &>/dev/null; then
    echo -e "${RED}Error: This script requires sudo privileges${NC}"
    exit 1
fi

# Detect script directory and set APP_DIR if running from deploy folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */deploy ]]; then
    APP_DIR="$(dirname "$SCRIPT_DIR")"
    echo -e "${GREEN}Detected APP_DIR: $APP_DIR${NC}"
fi

echo -e "${GREEN}Step 1: Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

echo ""
echo -e "${GREEN}Step 2: Installing Python ${PYTHON_VERSION}...${NC}"
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y ${PYTHON_VERSION} ${PYTHON_VERSION}-venv ${PYTHON_VERSION}-dev

echo ""
echo -e "${GREEN}Step 3: Installing system dependencies...${NC}"
sudo apt install -y \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev

echo ""
echo -e "${GREEN}Step 4: Hardening SSH security...${NC}"
# Install and configure fail2ban
sudo apt install -y fail2ban

# Create fail2ban jail for SSH
sudo tee /etc/fail2ban/jail.local > /dev/null << 'EOF'
[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
findtime = 600
EOF

# Ensure SSH is configured securely
sudo tee /etc/ssh/sshd_config.d/hardening.conf > /dev/null << 'EOF'
# Disable password authentication (use SSH keys only)
PasswordAuthentication no
ChallengeResponseAuthentication no

# Disable root login
PermitRootLogin no

# Only allow SSH protocol 2
Protocol 2

# Limit authentication attempts
MaxAuthTries 3

# Disconnect idle sessions after 10 minutes
ClientAliveInterval 300
ClientAliveCountMax 2

# Disable X11 forwarding (not needed)
X11Forwarding no

# Disable TCP forwarding for security (except we need local port forwarding for tunnel)
AllowTcpForwarding local
EOF

# Restart services
sudo systemctl restart fail2ban
sudo systemctl restart sshd

echo -e "${GREEN}SSH hardening complete:${NC}"
echo "  - Password authentication: disabled"
echo "  - Root login: disabled"
echo "  - fail2ban: enabled (3 failed attempts = 1 hour ban)"

echo ""
echo -e "${GREEN}Step 5: Installing MongoDB...${NC}"
# Check if MongoDB is already installed
if ! command -v mongod &> /dev/null; then
    curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
        sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
        sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    sudo apt update
    sudo apt install -y mongodb-org
    sudo systemctl start mongod
    sudo systemctl enable mongod
    echo -e "${GREEN}MongoDB installed and started${NC}"
else
    echo -e "${YELLOW}MongoDB already installed${NC}"
    sudo systemctl start mongod || true
fi

echo ""
echo -e "${GREEN}Step 6: Setting up application directory...${NC}"
# Ensure app directory exists and has correct permissions
if [ ! -d "$APP_DIR" ]; then
    sudo mkdir -p "$APP_DIR"
fi
sudo chown -R $USER:$USER "$APP_DIR"

# Create logs directory
mkdir -p "$LOG_DIR"

echo ""
echo -e "${GREEN}Step 7: Creating Python virtual environment...${NC}"
cd "$APP_DIR"
${PYTHON_VERSION} -m venv venv
source venv/bin/activate

echo ""
echo -e "${GREEN}Step 8: Installing Python dependencies...${NC}"
pip install --upgrade pip wheel setuptools

# Install live_trading requirements (this is all the server needs)
if [ -f "$APP_DIR/live_trading/requirements.txt" ]; then
    pip install -r "$APP_DIR/live_trading/requirements.txt"
else
    echo -e "${RED}live_trading/requirements.txt not found!${NC}"
    exit 1
fi

# Optionally install root requirements (ML/backtesting - not needed for live trading)
# These are heavy packages (keras, xgboost, etc.) only needed for backtesting
if [ -f "$APP_DIR/requirements.txt" ]; then
    read -p "Install ML/backtesting packages from root requirements.txt? (not needed for live trading) [y/N]: " INSTALL_ML
    INSTALL_ML=${INSTALL_ML:-N}
    if [[ "$INSTALL_ML" =~ ^[Yy]$ ]]; then
        grep -v "^-e \." "$APP_DIR/requirements.txt" > /tmp/requirements_filtered.txt || true
        pip install -r /tmp/requirements_filtered.txt || echo -e "${YELLOW}Some ML packages failed to install (this may be OK)${NC}"
    else
        echo -e "${GREEN}Skipping ML/backtesting packages${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Step 9: Setting up environment configuration...${NC}"
# Create .env file if it doesn't exist
if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$APP_DIR/deploy/env.example" ]; then
        cp "$APP_DIR/deploy/env.example" "$APP_DIR/.env"
        echo -e "${YELLOW}Created .env from template. Please edit it with your credentials:${NC}"
        echo -e "${YELLOW}  sudo nano $APP_DIR/.env${NC}"
    else
        echo -e "${YELLOW}No .env.example found. Please create $APP_DIR/.env manually${NC}"
    fi
else
    echo -e "${GREEN}.env file already exists${NC}"
fi

# Secure the .env file
chmod 600 "$APP_DIR/.env" 2>/dev/null || true

echo ""
echo -e "${GREEN}Step 10: Installing systemd service...${NC}"
# Get current username
CURRENT_USER=$(whoami)

# Create systemd service file with correct user
sudo tee /etc/systemd/system/trading-bot.service > /dev/null << EOF
[Unit]
Description=Trading Bot Backend
After=network.target mongod.service
Wants=mongod.service

[Service]
Type=simple
User=${CURRENT_USER}
Group=${CURRENT_USER}
WorkingDirectory=${APP_DIR}
Environment="PATH=${APP_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/venv/bin/python -m live_trading.main --daemon --log-dir=${LOG_DIR}
Restart=always
RestartSec=10
StandardOutput=append:${LOG_DIR}/daemon_stdout.log
StandardError=append:${LOG_DIR}/daemon_stderr.log

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=${APP_DIR}/logs ${LOG_DIR}
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable trading-bot

echo ""
echo -e "${GREEN}Step 11: Creating health check script...${NC}"
mkdir -p "$APP_DIR/scripts"
cat > "$APP_DIR/scripts/health_check.sh" << 'EOF'
#!/bin/bash
# Health check script - restarts service if API is unresponsive

HEALTH_URL="http://localhost:8000/api/health"
LOG_FILE="/opt/trading-bot/logs/health_check.log"

response=$(curl -sf -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null)

if [ "$response" != "200" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Health check failed (HTTP $response), restarting service..." >> "$LOG_FILE"
    sudo systemctl restart trading-bot
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Health check OK" >> "$LOG_FILE"
fi
EOF
chmod +x "$APP_DIR/scripts/health_check.sh"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo ""
echo "1. Edit your environment configuration:"
echo -e "   ${YELLOW}nano $APP_DIR/.env${NC}"
echo ""
echo "2. Start the trading bot service:"
echo -e "   ${YELLOW}sudo systemctl start trading-bot${NC}"
echo ""
echo "3. Check service status:"
echo -e "   ${YELLOW}sudo systemctl status trading-bot${NC}"
echo ""
echo "4. View logs:"
echo -e "   ${YELLOW}journalctl -u trading-bot -f${NC}"
echo -e "   ${YELLOW}tail -f $LOG_DIR/live_trading.log${NC}"
echo ""
echo "5. (Optional) Add health check to crontab:"
echo -e "   ${YELLOW}crontab -e${NC}"
echo "   Add: */5 * * * * $APP_DIR/scripts/health_check.sh"
echo ""
echo -e "${GREEN}From your local machine, connect via SSH tunnel:${NC}"
echo -e "   ${YELLOW}gcloud compute ssh algo-trading-bot --zone=europe-west2-c -- -L 8000:localhost:8000 -N${NC}"
echo ""
