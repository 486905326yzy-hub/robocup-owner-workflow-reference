#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-wpb-owner-search:noetic}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-full}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$MODE" in
  full)
    LAUNCH_FILE="task1_owner_search_real.launch"
    CONTAINER_NAME="wpb-owner-search-full"
    ;;
  bringup)
    LAUNCH_FILE="task1_owner_search_bringup.launch"
    CONTAINER_NAME="wpb-owner-search-bringup"
    ;;
  task)
    LAUNCH_FILE="task1_owner_search_task_only.launch"
    CONTAINER_NAME="wpb-owner-search-task"
    ;;
  enroll)
    LAUNCH_FILE="owner_enrollment_real.launch"
    CONTAINER_NAME="wpb-owner-enrollment"
    ;;
  shell)
    LAUNCH_FILE=""
    CONTAINER_NAME="wpb-owner-search-shell"
    ;;
  *)
    printf 'Usage: %s [full|bringup|task|enroll|shell] [roslaunch args...]\n' "$0" >&2
    exit 2
    ;;
esac

if command -v xhost >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

GPU_ARGS=()
if docker run --help 2>/dev/null | grep -q -- '--gpus'; then
  GPU_ARGS=(--gpus all)
fi

DEVICE_ARGS=()
if [ "$MODE" != "shell" ]; then
  KINECT_NODE="$({ lsusb -d 045e:02c4 || true; } | awk '{gsub(":", "", $2); gsub(":", "", $4); printf "/dev/bus/usb/%s/%s\n", $2, $4}')"
  if [ -z "$KINECT_NODE" ] || [ ! -e "$KINECT_NODE" ]; then
    echo "Kinect v2 not found; refusing to start the robot routine." >&2
    exit 1
  fi
  REQUIRED_DEVICES=(/dev/snd)
  if [ "$MODE" != "enroll" ]; then
    REQUIRED_DEVICES+=(/dev/ftdi /dev/rplidar)
  fi
  for DEVICE in "${REQUIRED_DEVICES[@]}"; do
    if [ ! -e "$DEVICE" ]; then
      echo "Required device not found: $DEVICE" >&2
      exit 1
    fi
  done
  DEVICE_ARGS=(--device="$KINECT_NODE":"$KINECT_NODE" --device=/dev/snd:/dev/snd)
  if [ "$MODE" != "enroll" ]; then
    DEVICE_ARGS+=(--device=/dev/ftdi:/dev/ftdi --device=/dev/rplidar:/dev/rplidar)
  fi
fi

COMMON_ARGS=(
  --rm
  -it
  --name "$CONTAINER_NAME"
  --network host
  --ipc host
  --user root
  -e DISPLAY="${DISPLAY:-}"
  -e QT_X11_NO_MITSHM=1
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e START_OLLAMA=1
  -e OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-999}"
  -e ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  -e ROS_IP="${ROS_IP:-}"
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  -v /etc/localtime:/etc/localtime:ro
  -v "$ROOT_DIR/catkin_ws:/home/ubuntu20/catkin_ws"
  -v "$ROOT_DIR/home_cache/insightface:/home/ubuntu20/.insightface"
  -v "$ROOT_DIR/home_cache/weights:/home/ubuntu20/weights"
  -v "$ROOT_DIR/waypoints.xml:/home/ubuntu20/waypoints.xml:ro"
)

if [ -d "/run/user/$(id -u)/pulse" ]; then
  COMMON_ARGS+=(-v "/run/user/$(id -u)/pulse:/run/user/1000/pulse" -e PULSE_SERVER="unix:/run/user/1000/pulse/native")
fi

if [ "$MODE" = "shell" ]; then
  exec docker run "${GPU_ARGS[@]}" "${COMMON_ARGS[@]}" "$IMAGE_NAME" bash
fi

if [ ! -f "$ROOT_DIR/catkin_ws/devel/setup.bash" ]; then
  echo "Workspace has not been compiled for the mounted source tree." >&2
  echo "Run: $ROOT_DIR/scripts/build_workspace.sh" >&2
  exit 1
fi

exec docker run "${GPU_ARGS[@]}" "${COMMON_ARGS[@]}" "${DEVICE_ARGS[@]}" "$IMAGE_NAME" \
  bash -lc "source /opt/ros/noetic/setup.bash && source /home/ubuntu20/catkin_ws/devel/setup.bash && roslaunch wpb_task1_owner_search ${LAUNCH_FILE} $*"
