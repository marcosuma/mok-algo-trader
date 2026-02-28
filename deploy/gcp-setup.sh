#!/bin/bash
#
# GCP Instance Setup Script for Trading Bot
# Run this from your local machine
#

set -e

# Configuration
INSTANCE_NAME="${INSTANCE_NAME:-algo-trading-bot}"
ZONE="${ZONE:-europe-west2-c}"  # London, UK
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-20GB}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Trading Bot - GCP Instance Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    echo "  macOS: brew install google-cloud-sdk"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${YELLOW}Not authenticated. Running gcloud auth login...${NC}"
    gcloud auth login
fi

# Get current project
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo -e "${YELLOW}No project set. Available projects:${NC}"
    gcloud projects list
    echo ""
    read -p "Enter project ID: " PROJECT
    gcloud config set project "$PROJECT"
fi

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Project:      $PROJECT"
echo "  Instance:     $INSTANCE_NAME"
echo "  Zone:         $ZONE"
echo "  Machine Type: $MACHINE_TYPE"
echo "  Disk Size:    $BOOT_DISK_SIZE"
echo ""

# Check if instance already exists
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; then
    echo -e "${YELLOW}Instance '$INSTANCE_NAME' already exists in zone '$ZONE'${NC}"
    read -p "Do you want to delete and recreate it? (y/N): " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        echo "Deleting existing instance..."
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
    else
        echo "Aborting. Use a different INSTANCE_NAME or zone."
        exit 1
    fi
fi

# Get user's current IP for SSH restriction
echo -e "${GREEN}Detecting your current IP address...${NC}"
MY_IP=$(curl -s ifconfig.me)
if [ -z "$MY_IP" ]; then
    echo -e "${RED}Could not detect your IP. Please enter it manually:${NC}"
    read -p "Your IP address: " MY_IP
fi
echo "Your IP: $MY_IP"
echo ""

read -p "Restrict SSH access to this IP only? (recommended) [Y/n]: " RESTRICT_SSH
RESTRICT_SSH=${RESTRICT_SSH:-Y}

echo -e "${GREEN}Creating GCP Compute Engine instance...${NC}"
echo ""

gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-standard \
    --metadata=enable-oslogin=TRUE,startup-script='#!/bin/bash
apt-get update -qq
apt-get install -y -qq git fail2ban' \
    --tags=trading-bot-server

# Create firewall rule to restrict SSH to user's IP
if [[ "$RESTRICT_SSH" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}Creating firewall rule to restrict SSH to your IP ($MY_IP)...${NC}"

    # Check if rule already exists
    if gcloud compute firewall-rules describe allow-ssh-from-home &>/dev/null; then
        echo "Updating existing firewall rule..."
        gcloud compute firewall-rules update allow-ssh-from-home \
            --source-ranges="$MY_IP/32"
    else
        # First, we need to remove the default SSH rule for our instance
        # Create a restrictive SSH rule just for our instance
        gcloud compute firewall-rules create allow-ssh-from-home \
            --direction=INGRESS \
            --network=default \
            --action=ALLOW \
            --rules=tcp:22 \
            --source-ranges="$MY_IP/32" \
            --target-tags=trading-bot-server \
            --priority=100 \
            --description="Allow SSH only from home IP for trading-bot"
    fi

    echo ""
    echo -e "${YELLOW}IMPORTANT: SSH is now restricted to IP: $MY_IP${NC}"
    echo -e "${YELLOW}If your IP changes, update the firewall rule:${NC}"
    echo -e "${YELLOW}  gcloud compute firewall-rules update allow-ssh-from-home --source-ranges=NEW_IP/32${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Instance Created Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get instance external IP
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo -e "${GREEN}Instance Details:${NC}"
echo "  Name:        $INSTANCE_NAME"
echo "  Zone:        $ZONE"
echo "  External IP: $EXTERNAL_IP"
echo ""
echo -e "${GREEN}Security Features Enabled:${NC}"
echo "  - GCP OS Login (uses your Google account for SSH)"
echo "  - fail2ban (blocks brute force attempts)"
if [[ "$RESTRICT_SSH" =~ ^[Yy]$ ]]; then
echo "  - SSH restricted to IP: $MY_IP"
fi
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo ""
echo "1. SSH into the instance:"
echo -e "   ${YELLOW}gcloud compute ssh $INSTANCE_NAME --zone=$ZONE${NC}"
echo ""
echo "2. Clone your repository and run install script:"
echo -e "   ${YELLOW}git clone YOUR_REPO_URL /opt/trading-bot${NC}"
echo -e "   ${YELLOW}cd /opt/trading-bot/deploy && chmod +x install.sh && ./install.sh${NC}"
echo ""
echo "3. Configure environment variables:"
echo -e "   ${YELLOW}sudo nano /opt/trading-bot/.env${NC}"
echo ""
echo "4. Start the service:"
echo -e "   ${YELLOW}sudo systemctl start trading-bot${NC}"
echo ""
echo "5. From your local machine, open SSH tunnel:"
echo -e "   ${YELLOW}gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -L 8000:localhost:8000 -N${NC}"
echo ""
echo "6. Access frontend at http://localhost:3000 (run npm run dev locally)"
echo ""

# Add convenience alias suggestion
echo -e "${GREEN}Tip: Add these aliases to your ~/.zshrc:${NC}"
echo ""
cat << 'ALIASES'
alias trading-tunnel='gcloud compute ssh algo-trading-bot --zone=europe-west2-c -- -L 8000:localhost:8000 -N'
alias trading-ssh='gcloud compute ssh algo-trading-bot --zone=europe-west2-c'
alias trading-logs='gcloud compute ssh algo-trading-bot --zone=europe-west2-c -- journalctl -u trading-bot -f'
ALIASES
echo ""

if [[ "$RESTRICT_SSH" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}If your home IP changes, run:${NC}"
    echo -e "${YELLOW}  NEW_IP=\$(curl -s ifconfig.me) && gcloud compute firewall-rules update allow-ssh-from-home --source-ranges=\$NEW_IP/32${NC}"
    echo ""
fi
