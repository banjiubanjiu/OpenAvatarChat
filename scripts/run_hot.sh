#!/usr/bin/env bash
# 简单热重载启动脚本，默认监控 src 和 config 目录。
# 用法：bash scripts/run_hot.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python scripts/run_with_reload.py -- uv run src/demo.py --config config/chat_with_lam_qwen3_asr.yaml
