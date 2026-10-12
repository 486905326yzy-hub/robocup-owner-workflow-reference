#!/usr/bin/env bash
set -euo pipefail

# Compile the locally mounted ROS workspace with the stable Docker environment.
# Run this after changes that need catkin to refresh generated files.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-wpb-owner-search:noetic}"

exec docker run --rm -it --network host --user root \
  -v "$ROOT_DIR/catkin_ws:/home/ubuntu20/catkin_ws" \
  -v "$ROOT_DIR/home_cache/insightface:/home/ubuntu20/.insightface" \
  -v "$ROOT_DIR/home_cache/weights:/home/ubuntu20/weights" \
  -v "$ROOT_DIR/waypoints.xml:/home/ubuntu20/waypoints.xml:ro" \
  "$IMAGE_NAME" \
  bash -lc 'source /opt/ros/noetic/setup.bash && cd /home/ubuntu20/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release'
