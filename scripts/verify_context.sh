#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required=(
  "$ROOT_DIR/Dockerfile"
  "$ROOT_DIR/catkin_ws/src/wpb_task1_owner_search/package.xml"
  "$ROOT_DIR/catkin_ws/src/wpb_task1_owner_search/scripts/task1_find_owner_real.py"
  "$ROOT_DIR/catkin_ws/src/wpb_task1_owner_search/launch/task1_owner_search_real.launch"
  "$ROOT_DIR/catkin_ws/src/wpb_home/wpb_home_bringup/package.xml"
  "$ROOT_DIR/catkin_ws/src/iai_kinect2/kinect2_bridge/package.xml"
  "$ROOT_DIR/catkin_ws/src/yoloworld_perception/models/yolov8s-world-person.pt"
  "$ROOT_DIR/catkin_ws/src/wpr_simulation/models/vision/yolo11n-pose.pt"
  "$ROOT_DIR/home_cache/insightface/models/buffalo_sc/det_500m.onnx"
  "$ROOT_DIR/waypoints.xml"
)

missing=0
for path in "${required[@]}"; do
  if [ ! -e "$path" ]; then
    printf 'MISSING %s\n' "$path"
    missing=1
  else
    printf 'OK      %s\n' "$path"
  fi
done

du -sh "$ROOT_DIR"
exit "$missing"
