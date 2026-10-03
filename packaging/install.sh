#!/usr/bin/env bash
# Installs transcribe to /opt/transcribe as a root systemd service that is
# NOT started at boot.  Usage:  sudo packaging/install.sh
# Re-run to update /opt/transcribe to the current checkout; your
# /etc/default/transcribe is never overwritten.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/opt/transcribe
STATE=/var/lib/transcribe

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required (apt install ffmpeg)" >&2; exit 1; }

# uv: prefer the invoking user's copy so we reuse their download cache.
USER_HOME=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)
UV=$(command -v uv || true)
[[ -z $UV && -x $USER_HOME/.local/bin/uv ]] && UV=$USER_HOME/.local/bin/uv
[[ -n $UV ]] || { echo "uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }

echo "==> copying code to $DEST"
mkdir -p "$DEST"
cp -r "$SRC/pyproject.toml" "$SRC/uv.lock" "$SRC/README.md" "$DEST/"
rm -rf "$DEST/transcribe" "$DEST/packaging"
cp -r "$SRC/transcribe" "$SRC/packaging" "$DEST/"
find "$DEST" -name __pycache__ -prune -exec rm -rf {} +

echo "==> building virtualenv"
# Build as the invoking user so their uv cache is used without leaving
# root-owned files in it, then hand everything to root. The explicit system
# interpreter keeps the venv independent of anything under the user's home.
if [[ -n ${SUDO_USER:-} ]]; then
    chown -R "$SUDO_USER" "$DEST"
    sudo -u "$SUDO_USER" -H "$UV" sync --frozen --no-dev --project "$DEST" --python /usr/bin/python3.12
else
    "$UV" sync --frozen --no-dev --project "$DEST" --python /usr/bin/python3.12
fi
chown -R root:root "$DEST"

mkdir -p "$STATE/jobs" "$STATE/huggingface" "$STATE/torch"

if [[ ! -f /etc/default/transcribe ]]; then
    echo "==> writing /etc/default/transcribe"
    install -m 600 "$SRC/packaging/transcribe.default" /etc/default/transcribe
fi

install -m 644 "$SRC/packaging/transcribe.service" /etc/systemd/system/transcribe.service
systemctl daemon-reload

echo
echo "Installed. The service does not start at boot. Use:"
echo "  sudo service transcribe start | stop | restart | status"
echo "Config: /etc/default/transcribe   Logs: journalctl -u transcribe -f"
