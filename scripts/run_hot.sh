#!/usr/bin/env bash
# 简单热重载启动脚本，默认监控 src 和 config 目录。
# 用法：bash scripts/run_hot.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 允许通过第一个参数指定配置文件，
# 默认使用翻译模式配置，如需原始对话模式可执行：
#   bash scripts/run_hot.sh config/chat_with_lam_qwen3_asr.yaml
CONFIG_PATH="${1:-config/chat_with_lam_qwen3_asr_translate.yaml}"

python scripts/run_with_reload.py -- uv run src/demo.py --config "${CONFIG_PATH}"
