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

Video processing is available over HTTP. Browser camera access requires a secure context, so use HTTPS with a real domain for remote real-time camera detection; `localhost` development access is also accepted by browsers.

## HTTPS

Point the domain's A record to the server before enabling HTTPS. In the cloud security group, allow TCP 443 in addition to TCP 80. The following command obtains a Let's Encrypt certificate for both the primary domain and its alias, changes Nginx to HTTPS, redirects HTTP traffic, and enables certificate renewal.

```bash
cd /opt/traffic_detection_core
sudo bash deploy/cpu-server/enable-https.sh your-email@example.com dnsgo.xyz www.dnsgo.xyz
```

If TCP 443 is already occupied by another service, the script automatically selects TCP 8443 and prints the port in its final message. Open the printed HTTPS URL, such as `https://dnsgo.xyz:8443`. Add the selected port to the cloud security group. Later project updates should use the domain names rather than the public IP so that the HTTPS configuration and trusted host settings are retained:

```bash
bash deploy/cpu-server/deploy.sh dnsgo.xyz www.dnsgo.xyz
```

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
