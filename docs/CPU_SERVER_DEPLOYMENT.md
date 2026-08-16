# CPU Server Deployment

This deployment targets a small Ubuntu server without an NVIDIA GPU. It is suitable for the graduation-project demo: dashboard access, model switching, and occasional image detection. Do not run model training or concurrent long video jobs on a 2-core, 2GB server.

The deployment runs the API in Docker on `127.0.0.1:8000`. Existing Nginx serves the frontend and proxies API, WebSocket, image, and video paths. Runtime data remains in Docker named volumes.

## Prerequisites

On Ubuntu, install Git, Docker Engine with the Compose plugin, Nginx, and OpenSSL. Configure the cloud security group to allow TCP 80. The backend port 8000 stays private and must not be opened in the security group.

The server must already be able to clone the repository through its GitHub SSH key:

```bash
git clone git@github.com:fastbit-ljb/traffic_detection_core.git /opt/traffic_detection_core
cd /opt/traffic_detection_core
```

For a 2GB server, create a 2GB swap file before the first image build:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Deploy

Run the deployment script with the public IP or domain. The first run creates the production environment file, generates a JWT secret, builds the CPU API image, builds frontend static files, and installs the Nginx site.

```bash
cd /opt/traffic_detection_core
bash deploy/cpu-server/deploy.sh 43.108.38.230
```

Open `http://43.108.38.230` after the command reports success.

## Update

```bash
cd /opt/traffic_detection_core
git pull --ff-only
bash deploy/cpu-server/deploy.sh 43.108.38.230
```

## Operations

```bash
docker compose -f deploy/cpu-server/docker-compose.cpu.yml ps
docker compose -f deploy/cpu-server/docker-compose.cpu.yml logs -f api
docker compose -f deploy/cpu-server/docker-compose.cpu.yml restart api
```

The first API image build downloads the three official YOLOv8 weights (`n`, `s`, and `m`) and packages the representative self-trained traffic weight. The deployment script copies all four into the persistent `traffic-models` volume before the API starts. Detection history, uploaded files, generated images, generated videos, logs, and models are persistent Docker volumes. List them with `docker volume ls | grep traffic`.
