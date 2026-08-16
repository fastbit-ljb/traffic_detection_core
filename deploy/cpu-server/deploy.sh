#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash deploy/cpu-server/deploy.sh <public-ip-or-domain>"
  exit 1
fi

server_name="$1"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
deploy_dir="$project_dir/deploy/cpu-server"
env_file="$deploy_dir/.env.production"
site_root="/var/www/traffic-detection"
nginx_site="/etc/nginx/sites-available/traffic-detection"
public_origin="http://$server_name"

command -v docker >/dev/null || { echo "Docker is required."; exit 1; }
docker compose version >/dev/null || { echo "Docker Compose plugin is required."; exit 1; }
command -v nginx >/dev/null || { echo "Nginx is required."; exit 1; }

if [[ ! -f "$env_file" ]]; then
  secret="$(openssl rand -hex 32)"
  sed \
    -e "s|__PUBLIC_ORIGIN__|$public_origin|" \
    -e "s|__GENERATE_ON_DEPLOY__|$secret|" \
    "$deploy_dir/.env.production.template" > "$env_file"
  chmod 600 "$env_file"
  echo "Created $env_file. Keep this file private."
fi

docker compose --env-file "$env_file" -f "$deploy_dir/docker-compose.cpu.yml" up -d --build

frontend_image="traffic-detection-frontend-build"
frontend_container="traffic-detection-frontend-export"
docker build -f "$deploy_dir/Dockerfile.frontend" -t "$frontend_image" "$project_dir"
docker rm -f "$frontend_container" >/dev/null 2>&1 || true
docker create --name "$frontend_container" "$frontend_image" >/dev/null
sudo install -d -m 755 "$site_root"
sudo rm -rf "$site_root"/*
docker cp "$frontend_container:/usr/share/nginx/html/." "$site_root"
docker rm "$frontend_container" >/dev/null

sed "s|__SERVER_NAME__|$server_name|g" "$deploy_dir/nginx-traffic-detection.conf.template" | \
  sudo tee "$nginx_site" >/dev/null
sudo ln -sfn "$nginx_site" /etc/nginx/sites-enabled/traffic-detection
sudo nginx -t
sudo systemctl reload nginx

curl --fail --retry 10 --retry-delay 2 "http://127.0.0.1:8000/health" >/dev/null
echo "Deployment complete: $public_origin"
