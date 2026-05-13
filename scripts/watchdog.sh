#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/LLMconfig}"
SERVER_PORT="${SERVER_PORT:-8000}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3-aigc-chat:latest}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/watchdog.log}"
PID_FILE="${PID_FILE:-$PROJECT_DIR/watchdog.pid}"

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

log() {
    mkdir -p "$(dirname "$LOG_FILE")"
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG_FILE"
}

is_loop_running() {
    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi

    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -z "$pid" ]; then
        return 1
    fi

    if kill -0 "$pid" >/dev/null 2>&1 && ps -p "$pid" -o args= | grep -q "watchdog.sh loop"; then
        return 0
    fi

    rm -f "$PID_FILE"
    return 1
}

port_pids() {
    ss -ltnp 2>/dev/null \
        | sed -n "s/.*:${SERVER_PORT}.*pid=\([0-9][0-9]*\).*/\1/p" \
        | sort -u
}

start_ollama() {
    if curl -fsS --connect-timeout 3 --max-time 5 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
        return
    fi

    if pgrep -x ollama >/dev/null 2>&1; then
        log "killing unhealthy ollama"
        pkill -x ollama >/dev/null 2>&1 || true
        sleep 1
    fi

    log "starting ollama"
    nohup ollama serve >> /root/ollama.log 2>&1 &

    for _ in $(seq 1 20); do
        if curl -fsS --connect-timeout 3 --max-time 5 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
            log "ollama is healthy"
            return
        fi
        sleep 1
    done

    log "ollama did not become healthy"
}

start_api() {
    if curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1:${SERVER_PORT}/" >/dev/null 2>&1; then
        return
    fi

    local pids
    pids="$(port_pids || true)"
    if [ -n "$pids" ]; then
        log "killing unhealthy api pids: $pids"
        kill $pids >/dev/null 2>&1 || true
        sleep 1
    fi

    cd "$PROJECT_DIR"
    log "starting api on port $SERVER_PORT with model $OLLAMA_MODEL"
    OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_MODEL="$OLLAMA_MODEL" \
        OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE" SERVER_PORT="$SERVER_PORT" \
        nohup uv run aigc_rewriter_server.py >> server.log 2>&1 &

    for _ in $(seq 1 30); do
        if curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1:${SERVER_PORT}/" >/dev/null 2>&1; then
            log "api is healthy"
            return
        fi
        sleep 1
    done

    log "api did not become healthy"
}

start_nginx() {
    if curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1/" >/dev/null 2>&1; then
        return
    fi

    log "starting nginx"
    if command -v service >/dev/null 2>&1; then
        service nginx restart >> "$LOG_FILE" 2>&1 || nginx >> "$LOG_FILE" 2>&1 || true
    else
        nginx >> "$LOG_FILE" 2>&1 || true
    fi

    if curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1/" >/dev/null 2>&1; then
        log "nginx is healthy"
    else
        log "nginx did not become healthy"
    fi
}

check_once() {
    start_ollama
    start_api
    start_nginx
}

install_loop() {
    if is_loop_running; then
        log "watchdog loop already running"
        return
    fi

    log "starting watchdog loop"
    nohup "$0" loop >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
}

case "${1:-once}" in
    once)
        check_once
        ;;
    loop)
        echo $$ > "$PID_FILE"
        log "watchdog loop started, interval=${CHECK_INTERVAL}s"
        while true; do
            check_once
            sleep "$CHECK_INTERVAL"
        done
        ;;
    install)
        install_loop
        check_once
        ;;
    *)
        echo "Usage: $0 [once|loop|install]" >&2
        exit 2
        ;;
esac
