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

install -d -m 755 /etc/letsencrypt/renewal-hooks/deploy
cat > /etc/letsencrypt/renewal-hooks/deploy/reload-traffic-nginx <<'EOF'
#!/usr/bin/env bash
systemctl reload nginx
EOF
chmod 755 /etc/letsencrypt/renewal-hooks/deploy/reload-traffic-nginx
systemctl enable --now certbot.timer
bash "$deploy_script" "${domains[@]}"
echo "HTTPS enabled: https://$primary_domain"
