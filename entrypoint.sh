#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
if [ -f /home/ubuntu20/catkin_ws/devel/setup.bash ]; then
  source /home/ubuntu20/catkin_ws/devel/setup.bash
fi

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
if [ -n "${ROS_IP:-}" ]; then
  export ROS_IP
else
  unset ROS_IP
fi

# The owner-search task uses local Qwen only for interpreting the electrical
# switch instruction. Start Ollama when requested; model files are baked into
# /opt/ollama/models by the Dockerfile.
if [ "${START_OLLAMA:-0}" = "1" ]; then
  export OLLAMA_MODELS="${OLLAMA_MODELS:-/opt/ollama/models}"
  ollama serve >/tmp/ollama.log 2>&1 &
  ollama_pid=$!
  for attempt in $(seq 1 30); do
    if ollama list >/dev/null 2>&1; then break; fi
    sleep 1
  done
  if ! ollama list >/dev/null 2>&1; then
    echo "Ollama did not become ready; continuing without Qwen." >&2
  fi
  trap 'kill "${ollama_pid}" 2>/dev/null || true' EXIT
fi

exec "$@"
