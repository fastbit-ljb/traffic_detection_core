#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sudo bash deploy/cpu-server/enable-https.sh <email> <primary-domain> [domain-alias ...]"
  exit 1
fi

email="$1"
shift
domains=("$@")
primary_domain="${domains[0]}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
deploy_script="$project_dir/deploy/cpu-server/deploy.sh"
certificate_dir="/etc/letsencrypt/live/$primary_domain"

if [[ "$primary_domain" =~ ^[0-9.]+$ ]]; then
  echo "A domain name is required for HTTPS, not a public IP address."
  exit 1
fi

if [[ ! -f "$certificate_dir/fullchain.pem" ]]; then
  bash "$deploy_script" "${domains[@]}"
  apt-get update
  apt-get install -y certbot
  certbot_args=(certonly --webroot -w /var/www/certbot --email "$email" --agree-tos --no-eff-email)
  for domain in "${domains[@]}"; do
    certbot_args+=(-d "$domain")
  done
  certbot "${certbot_args[@]}"
fi

env_file="$project_dir/deploy/cpu-server/.env.production"
configured_https_port="$(sed -n 's/^TRAFFIC_HTTPS_PORT=//p' "$env_file" | tail -n 1)"
if [[ "$configured_https_port" =~ ^[0-9]+$ ]]; then
  https_port="$configured_https_port"
  if [[ "$https_port" == "443" ]]; then
    listener="$(ss -ltnp 2>/dev/null | grep -E '(:|\])443[[:space:]]' || true)"
    if [[ -n "$listener" ]] && ! grep -q 'nginx' <<<"$listener"; then
      echo "TCP 443 is already in use by another service; selecting TCP 8443 for traffic detection HTTPS."
      https_port="8443"
    fi
  fi
else
  listener="$(ss -ltnp 2>/dev/null | grep -E '(:|\])443[[:space:]]' || true)"
  if [[ -n "$listener" ]] && ! grep -q 'nginx' <<<"$listener"; then
    echo "TCP 443 is already in use by another service; selecting TCP 8443 for traffic detection HTTPS."
    https_port="8443"
  else
    https_port="443"
  fi
fi
if grep -q '^TRAFFIC_HTTPS_PORT=' "$env_file"; then
  sed -i "s|^TRAFFIC_HTTPS_PORT=.*|TRAFFIC_HTTPS_PORT=$https_port|" "$env_file"
else
  printf '\nTRAFFIC_HTTPS_PORT=%s\n' "$https_port" >> "$env_file"
fi

install -d -m 755 /etc/letsencrypt/renewal-hooks/deploy
cat > /etc/letsencrypt/renewal-hooks/deploy/reload-traffic-nginx <<'EOF'
#!/usr/bin/env bash
systemctl reload nginx
EOF
chmod 755 /etc/letsencrypt/renewal-hooks/deploy/reload-traffic-nginx
systemctl enable --now certbot.timer
bash "$deploy_script" "${domains[@]}"
if [[ "$https_port" == "443" ]]; then
  echo "HTTPS enabled: https://$primary_domain"
else
  echo "HTTPS enabled: https://$primary_domain:$https_port"
fi
