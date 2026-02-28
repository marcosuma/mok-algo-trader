# GCP Deployment Guide

This guide covers deploying the trading bot backend to Google Cloud Platform (GCP) with a security-first approach using SSH tunnels.

## Architecture

```
┌─────────────────────┐         SSH Tunnel          ┌─────────────────────┐
│   Your Local Mac    │◄──────────────────────────►│    GCP VM (e2-micro) │
│                     │                             │                     │
│  ┌───────────────┐  │                             │  ┌───────────────┐  │
│  │   Frontend    │  │                             │  │   Backend API │  │
│  │  (localhost)  │  │                             │  │ (localhost:8000)│  │
│  └───────────────┘  │                             │  └───────────────┘  │
│         │           │                             │         │           │
│         ▼           │                             │         ▼           │
│   localhost:8000 ───┼─────────────────────────────┼──► localhost:8000   │
│                     │                             │         │           │
└─────────────────────┘                             │         ▼           │
                                                    │  ┌───────────────┐  │
                                                    │  │   MongoDB     │  │
                                                    │  │ (Atlas/Local) │  │
                                                    │  └───────────────┘  │
                                                    └─────────────────────┘
```

## Security Features

- **No public API exposure**: API binds to localhost only
- **SSH tunnel required**: All traffic encrypted through SSH
- **SSH restricted to your IP**: Firewall blocks all other IPs from port 22
- **SSH key-only auth**: Password login disabled, keys required
- **fail2ban enabled**: Auto-blocks IPs after 3 failed login attempts
- **GCP OS Login**: Uses your Google account for SSH authentication
- **Secrets isolation**: `.env` file with restricted permissions (chmod 600)

### If Your Home IP Changes

Your ISP may change your IP periodically. When this happens, run:

```bash
./deploy/update-ssh-ip.sh
```

Or manually:
```bash
NEW_IP=$(curl -s ifconfig.me)
gcloud compute firewall-rules update allow-ssh-from-home --source-ranges=$NEW_IP/32
```

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed locally:
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Or download from https://cloud.google.com/sdk/docs/install
   ```
3. **Authenticate gcloud**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

## Quick Start

### Step 1: Create GCP Instance (Run Locally)

```bash
cd deploy
chmod +x gcp-setup.sh
./gcp-setup.sh
```

This creates an e2-micro instance (~$6/month) in us-central1-a.

### Step 2: Install on Server

SSH into the instance:
```bash
gcloud compute ssh algo-trading-bot --zone=europe-west2-c
```

Clone and run the install script:
```bash
git clone YOUR_REPO_URL /opt/trading-bot
cd /opt/trading-bot/deploy
chmod +x install.sh
./install.sh
```

### Step 3: Configure Environment

Edit the `.env` file on the server:
```bash
sudo nano /opt/trading-bot/.env
```

Add your broker credentials (see `.env.example` for template).

### Step 4: Start the Service

```bash
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

### Step 5: Connect from Local Machine

Open SSH tunnel:
```bash
gcloud compute ssh algo-trading-bot --zone=europe-west2-c -- -L 8000:localhost:8000 -N
```

Then run frontend locally:
```bash
cd live_trading/frontend
npm run dev
```

Access dashboard at: http://localhost:3000

## Commands Reference

### Local Machine

| Command | Description |
|---------|-------------|
| `./deploy/gcp-setup.sh` | Create GCP instance |
| `gcloud compute ssh algo-trading-bot --zone=europe-west2-c` | SSH into server |
| `gcloud compute ssh algo-trading-bot --zone=europe-west2-c -- -L 8000:localhost:8000 -N` | Open SSH tunnel |
| `gcloud compute instances stop algo-trading-bot --zone=europe-west2-c` | Stop instance (save costs) |
| `gcloud compute instances start algo-trading-bot --zone=europe-west2-c` | Start instance |
| `gcloud compute instances delete algo-trading-bot --zone=europe-west2-c` | Delete instance |

### On GCP Server

| Command | Description |
|---------|-------------|
| `sudo systemctl start trading-bot` | Start the service |
| `sudo systemctl stop trading-bot` | Stop the service |
| `sudo systemctl restart trading-bot` | Restart the service |
| `sudo systemctl status trading-bot` | Check service status |
| `journalctl -u trading-bot -f` | Follow service logs |
| `tail -f /opt/trading-bot/logs/live_trading.log` | View application logs |

## Updating the Code

On the GCP server:
```bash
cd /opt/trading-bot
git pull origin main
sudo systemctl restart trading-bot
```

## MongoDB Options

### Option A: MongoDB Atlas (Recommended for Production)

1. Create free M0 cluster at https://cloud.mongodb.com
2. Whitelist `0.0.0.0/0` or your GCP instance's IP
3. Update `.env`:
   ```
   MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
   ```

### Option B: Local MongoDB on GCP

Already installed by `install.sh`. Data persists on the VM's disk.

## Troubleshooting

### Service Won't Start

```bash
# Check logs
journalctl -u trading-bot -n 100

# Check if MongoDB is running
sudo systemctl status mongod

# Test manually
cd /opt/trading-bot
source venv/bin/activate
python -m live_trading.main
```

### SSH Tunnel Disconnects

Use autossh for persistent tunnels:
```bash
# Install
brew install autossh

# Run (auto-reconnects)
autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" \
  -L 8000:localhost:8000 -N algo-trading-bot
```

### Check API Health

Through the tunnel:
```bash
curl http://localhost:8000/api/health
```

## Cost Optimization

- **Stop when not trading**: `gcloud compute instances stop algo-trading-bot --zone=europe-west2-c`
- **Use preemptible instance**: Add `--preemptible` flag (cheaper but may restart)
- **Committed use discounts**: For long-term use

## Files in This Directory

| File | Purpose |
|------|---------|
| `README.md` | This documentation |
| `gcp-setup.sh` | Creates GCP Compute Engine instance |
| `install.sh` | Server-side installation script |
| `tunnel.sh` | SSH tunnel helper script (run locally) |
| `update-ssh-ip.sh` | Update firewall when your IP changes |
| `trading-bot.service` | systemd service configuration |
| `env.example` | Environment variables template |
