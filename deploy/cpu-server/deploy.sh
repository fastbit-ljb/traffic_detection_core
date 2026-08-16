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

compose=(docker compose --env-file "$env_file" -f "$deploy_dir/docker-compose.cpu.yml")
"${compose[@]}" build api

# Named volumes can be created as root on a first deployment. Initialize every
# writable runtime directory as root once, then hand it to the non-root API user.
# This also repairs volumes created by earlier versions of the deployment image.
"${compose[@]}" run --rm --no-deps --user root api sh -ec '
  mkdir -p /app/storage /app/uploads /app/output_images /app/output_videos /app/models /app/logs
  for model in /opt/bootstrap-models/*.pt; do
    [ -s "$model" ] || continue
    target="/app/models/${model##*/}"
    [ -s "$target" ] || cp "$model" "$target"
  done
  chown -R traffic:traffic /app/storage /app/uploads /app/output_images /app/output_videos /app/models /app/logs
'
"${compose[@]}" up -d

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

# A just-created Uvicorn process can reset its first connection while loading.
# Retry every curl failure instead of treating that transient reset as a deploy failure.
health_response="$(curl --fail --retry 10 --retry-all-errors --retry-delay 2 "http://127.0.0.1:8000/health")"
if ! grep -q '"status":"healthy"' <<<"$health_response"; then
  echo "API started but its services are not ready: $health_response" >&2
  exit 1
fi
echo "Deployment complete: $public_origin"
