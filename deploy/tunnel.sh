#!/bin/bash
#
# SSH Tunnel Script for Trading Bot
# Run this on your local machine to connect to the GCP backend
#

# Configuration
INSTANCE_NAME="${INSTANCE_NAME:-algo-trading-bot}"
ZONE="${ZONE:-europe-west2-c}"  # London, UK
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Trading Bot - SSH Tunnel${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Instance: $INSTANCE_NAME"
echo "Zone:     $ZONE"
echo "Tunnel:   localhost:$LOCAL_PORT -> remote:$REMOTE_PORT"
echo ""
echo -e "${YELLOW}Tunnel is active. Press Ctrl+C to disconnect.${NC}"
echo -e "${YELLOW}You can now access the API at http://localhost:$LOCAL_PORT${NC}"
echo ""

# Check if autossh is available (for auto-reconnection)
if command -v autossh &> /dev/null; then
    echo -e "${GREEN}Using autossh for auto-reconnection...${NC}"
    autossh -M 0 \
        -o "ServerAliveInterval=30" \
        -o "ServerAliveCountMax=3" \
        -o "ExitOnForwardFailure=yes" \
        -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} \
        -N \
        ${INSTANCE_NAME}
else
    echo -e "${YELLOW}Tip: Install autossh for auto-reconnection: brew install autossh${NC}"
    echo ""
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" -- \
        -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} \
        -N \
        -o "ServerAliveInterval=30" \
        -o "ServerAliveCountMax=3"
fi
