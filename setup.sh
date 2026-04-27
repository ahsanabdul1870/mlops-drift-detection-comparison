#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"

log() {
  echo "[setup] $1"
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment at .venv"
    python3 -m venv "$VENV_DIR"
  else
    log "Virtual environment already exists"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip >/dev/null
}

requirements_hash() {
  sha256sum "$REQ_FILE" | awk '{print $1}'
}

ensure_requirements() {
  if [[ ! -f "$REQ_FILE" ]]; then
    log "requirements.txt not found. Skipping dependency installation."
    return
  fi

  local current_hash
  current_hash="$(requirements_hash)"
  local saved_hash=""

  if [[ -f "$REQ_HASH_FILE" ]]; then
    saved_hash="$(cat "$REQ_HASH_FILE")"
  fi

  local needs_install="false"

  if [[ "$current_hash" != "$saved_hash" ]]; then
    needs_install="true"
    log "Dependency checkpoint changed (or first run)."
  else
    if ! pip check >/dev/null 2>&1; then
      needs_install="true"
      log "Dependency health check failed. Re-installing requirements."
    fi
  fi

  if [[ "$needs_install" == "true" ]]; then
    log "Installing dependencies from requirements.txt"
    pip install -r "$REQ_FILE"
    pip check
    echo "$current_hash" > "$REQ_HASH_FILE"
    log "Dependencies installed and checkpoint updated"
  else
    log "Dependencies are already installed and healthy"
  fi
}

run_pipeline() {
  log "Generating drift datasets"
  python "$PROJECT_ROOT/src/drift_simulator.py"

  log "Training baseline model"
  python "$PROJECT_ROOT/src/train.py"

  log "Running drift detector experiments"
  python "$PROJECT_ROOT/src/drift_detector.py"

  log "Running tests"
  pytest "$PROJECT_ROOT/tests" -q
}

print_next_steps() {
  cat <<EOF

Setup complete.

Useful commands:
1) Activate venv:
   source .venv/bin/activate

2) Start API:
   uvicorn src.api:app --host 0.0.0.0 --port 8000

3) Start full Docker stack (API + Prometheus + Grafana):
   docker compose -f docker/docker-compose.yml up -d --build

4) Start MLflow UI:
   mlflow ui --backend-store-uri mlflow/mlruns --host 0.0.0.0 --port 5000
EOF
}

main() {
  cd "$PROJECT_ROOT"
  ensure_venv
  ensure_requirements
  run_pipeline
  print_next_steps
}

main "$@"
