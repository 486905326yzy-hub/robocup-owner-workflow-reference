#!/usr/bin/env bash
# Start the current full owner-registration and owner-search workflow.
# This script only launches existing containers and the already-built workspace;
# it never installs packages, downloads models, or changes the environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-wpb-owner-search:reid-runtime}"
MAIN_CONTAINER="wpb-owner-voice-reid"
KINECT_CONTAINER="wpb-kinect-camera"
DISPLAY_VALUE="${DISPLAY:-:1}"

if [ ! -f "$ROOT_DIR/catkin_ws/devel/setup.bash" ]; then
  echo "Workspace is not built: $ROOT_DIR/catkin_ws/devel/setup.bash" >&2
  exit 1
fi

BASE_PORT="${BASE_PORT:-/dev/ftdi}"
# Some machines do not load the old /dev/ftdi udev alias. Resolve the base
# through its stable FTDI USB identity instead of assuming a ttyUSB number.
if [ ! -e "$BASE_PORT" ]; then
  for port in /dev/serial/by-id/*FTDI*; do
    if [ -e "$port" ]; then
      BASE_PORT="$(readlink -f "$port")"
      echo "Using verified FTDI base port fallback: $BASE_PORT"
      break
    fi
  done
fi

for device in /dev/snd "$BASE_PORT" /dev/rplidar; do
  if [ ! -e "$device" ]; then
    echo "Required device not found: $device" >&2
    exit 1
  fi
done

KINECT_NODE="$({ lsusb -d 045e:02c4 2>/dev/null || true; } | awk '{gsub(":", "", $2); gsub(":", "", $4); printf "/dev/bus/usb/%s/%s", $2, $4}')"
if [ -z "$KINECT_NODE" ] || [ ! -e "$KINECT_NODE" ]; then
  echo "Kinect v2 (045e:02c4) not found; not starting." >&2
  exit 1
fi

if command -v xhost >/dev/null 2>&1; then
  DISPLAY="$DISPLAY_VALUE" xhost +local:docker >/dev/null 2>&1 || true
fi

# A restart is intentional here: remove only the two containers owned by this workflow.
docker rm -f "$MAIN_CONTAINER" "$KINECT_CONTAINER" >/dev/null 2>&1 || true

docker run -d --rm --name "$MAIN_CONTAINER" \
  --network host --ipc host --user root --gpus all \
  --device /dev/snd:/dev/snd --device "$BASE_PORT:$BASE_PORT" --device /dev/rplidar:/dev/rplidar \
  -e DISPLAY="$DISPLAY_VALUE" -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e START_OLLAMA=1 -e OLLAMA_NUM_GPU=999 -e BASE_PORT="$BASE_PORT" \
  -e ROS_MASTER_URI=http://localhost:11311 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/localtime:/etc/localtime:ro \
  -v "$ROOT_DIR/catkin_ws:/home/ubuntu20/catkin_ws" \
  -v "$ROOT_DIR/home_cache/insightface:/home/ubuntu20/.insightface" \
  -v "$ROOT_DIR/home_cache/weights:/home/ubuntu20/weights" \
  -v "$ROOT_DIR/waypoints.xml:/home/ubuntu20/waypoints.xml:ro" \
  "$IMAGE_NAME" bash -lc '
    source /opt/ros/noetic/setup.bash
    source /home/ubuntu20/catkin_ws/devel/setup.bash
    roslaunch wpb_task1_owner_search owner_voice_reid_yolo_pose.launch \
      serial_port:="$BASE_PORT" rplidar_port:=/dev/rplidar \
      start_camera:=false start_yolo:=true start_voice:=true \
      start_asr:=true start_tts:=true start_sound_play:=false \
      start_rviz:=true show_yolo_window:=true
  '

# The bridge is deliberately started after ROS master / RViz / task nodes are up.
sleep 6
docker run -d --rm --name "$KINECT_CONTAINER" \
  --privileged --network host --ipc host --user root \
  --device "$KINECT_NODE:$KINECT_NODE" \
  -e ROS_MASTER_URI=http://localhost:11311 \
  -v "$ROOT_DIR/catkin_ws:/home/ubuntu20/catkin_ws" \
  "$IMAGE_NAME" bash -lc '
    source /opt/ros/noetic/setup.bash
    source /home/ubuntu20/catkin_ws/devel/setup.bash
    roslaunch kinect2_bridge kinect2_bridge.launch fps_limit:=15 queue_size:=2 worker_threads:=2
  '

echo "Started: $MAIN_CONTAINER and $KINECT_CONTAINER"
echo "Logs: docker logs -f $MAIN_CONTAINER"
echo "Kinect: docker logs -f $KINECT_CONTAINER"
