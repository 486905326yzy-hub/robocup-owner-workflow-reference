# Robot project handoff — 2026-08-30

## Objective

Finish a safe end-to-end validation of the `wpb-owner-search:noetic` ROS Noetic robot project. The immediate blocker is the Kinect v2 visual pipeline: YOLO and point cloud nodes launch, but no Kinect image/point-cloud messages arrive in the container.

## Safety boundary (must follow)

- This is a real mobile robot. Do **not** launch a command that can move the base, run navigation, control the arm, or publish `/cmd_vel` unless the user explicitly reconfirms that the area is clear, an emergency stop is reachable, and an operator is supervising.
- Prefer no-motion tests: USB enumeration, read-only topic checks, standalone Kinect bridge tests, static RViz, source/config inspection, and Docker build/validation.
- Do not use `--privileged` or mount all of `/dev` merely as a workaround. Pass only the device(s) required by the test.
- Do not delete images, containers, source files, or removable-media backups unless the user explicitly asks.

## Host and project state

- Host: Ubuntu 24.04.4; NVIDIA RTX 5060; driver 595.84; Docker Engine + NVIDIA Container Toolkit working.
- Project directory: `/home/yanzy/wpb_owner_search`.
- Image: `wpb-owner-search:noetic` — successfully built.
- Base image: `osrf/ros:noetic-desktop-full` (Ubuntu 20.04 / ROS Noetic).
- GPU validation inside the image passed: PyTorch `2.4.1+cu118`, `torch.cuda.is_available() == True`.
- Docker daemon proxy works via `http://127.0.0.1:7897` when the user's proxy is running.

## Existing project changes

The project image was rebuilt successfully after these local changes:

1. `Dockerfile`: compatibility changes for Python 3.8 / pip (`tokenizers==0.20.3`, `psutil==7.2.2` with `--ignore-installed`) and late install of `ros-noetic-joy`.
2. `entrypoint.sh`: no longer exports an empty `ROS_IP`; it preserves a supplied value and otherwise unsets it. This fixed ROS errors such as `couldn't resolve publisher host []`.

Do not discard these changes.

## Confirmed hardware mappings and tests

Host udev rules in `/etc/udev/rules.d/99-wpb-robot.rules` create stable links:

| Device | Host identity | Stable link | Result |
| --- | --- | --- | --- |
| Base control board | FTDI `0403:6001`, serial `AB0LPLOE` | `/dev/ftdi` -> `ttyUSB0` | Driver starts; joint state positions are non-zero, so serial feedback is real. |
| RPLIDAR | CP2102 `10c4:ea60`, serial `0001` | `/dev/rplidar` -> `ttyUSB1` | Pass: A2M8, firmware 1.31, health OK, 115200 baud. |
| Kinect v2 | USB `045e:02c4`, Microsoft Xbox NUI Sensor | currently `Bus 004 Device 003` (dynamic) | Host sees it, but current Docker visual run does not open it. |

The following have already passed without moving the robot:

- Base core exposes `/odom`, `/imu/data_raw`, `/joint_states`; `/joint_states` contains non-zero arm/gripper positions.
- Lidar node starts and RViz displays live `/scan` over the map.
- The full navigation RViz configuration reaches `Global Status: Ok`, with `/map`, transforms and laser scan all visible.
- A prior standalone Kinect test did enumerate the device and reported a serial number, so this is likely a transient USB mapping/access issue rather than a missing dependency.

## Current visual-pipeline failure (most important)

The user launched the full task with narrowly passed devices. Logs show:

```text
[Info] [Freenect2Impl] 14 usb devices connected
[Info] [Freenect2Impl] found 0 devices
[ERROR] [Kinect2Bridge::initDevice] no Kinect2 devices found!
```

Consequences observed in the running container:

- Nodes exist: `/kinect2`, `/kinect2_bridge`, `/kinect2_points_xyzrgb_qhd`, `/yoloworld`, `/yoloworld_bbox_debug_viewer`.
- Topics exist: `/kinect2/qhd/image_color_rect`, `/kinect2/qhd/points`, `/perception/person_detections_2d`, `/perception/yoloworld/debug_image`.
- `rostopic hz /kinect2/qhd/image_color_rect` and `rostopic hz /kinect2/qhd/points` both report `no new messages`.
- Therefore no YOLO image window or 3D point cloud is expected yet. YOLO is installed and the node starts; it has no camera frames to process.

The previously used full-run command passed the whole USB bus as:

```bash
--device=/dev/bus/usb:/dev/bus/usb
```

But the bridge still found zero devices. First investigate with a **standalone, no-motion Kinect test** and current host device path. Confirm it afresh with:

```bash
lsusb -d 045e:02c4
```

Then prefer a specific device mapping (substitute the current bus/device values) rather than broad privileges:

```bash
sudo docker run --rm -it --network host --user root --gpus all \
  --device=/dev/bus/usb/004/003:/dev/bus/usb/004/003 \
  wpb-owner-search:noetic \
  bash -lc 'source /opt/ros/noetic/setup.bash; source /home/ubuntu20/catkin_ws/devel/setup.bash; roslaunch kinect2_bridge kinect2_bridge.launch'
```

This test must not receive `/dev/ftdi` or `/dev/rplidar`; it cannot move the robot. Diagnose from its actual output before changing Dockerfile or launch files. If the device number changes after reconnect, update the command.

## Additional issues after Kinect is fixed

### Face verification not ready

The task exited with:

```text
task1_find_owner_real failed: face verification is required but not ready; check owner_image_path and InsightFace model cache
```

Relevant paths:

- Config: `catkin_ws/src/wpb_task1_owner_search/config/task1_owner_search_real.yaml`
- Owner photos: `catkin_ws/src/wpb_task1_owner_search/data/owner` (9 JPEG files exist)
- InsightFace cache copied into image: `/home/ubuntu20/.insightface/models/buffalo_sc/`
- Cached files: `w600k_mbf.onnx` and `det_500m.onnx`.
- Config uses `face_model_name: buffalo_sc`, `face_auto_download: false`, `face_verify_required: true`.

Check the detailed warning emitted by `task1_find_owner_real.py`: it may be an invalid/no-face reference photo, model/provider initialization problem, or an image-loading issue. Do not solve it by silently setting `face_verify_required: false` unless the user expressly asks to relax the task's identity-verification requirement.

### Audio input

`arecord` failed to open ALSA capture. This affects speech recognition only; it is not the reason for missing Kinect/Yolo output. Treat it after the visual pipeline works.

## Safe diagnostic workflow

1. Ensure any prior full task is stopped (`Ctrl+C`) before standalone testing.
2. Verify host Kinect enumeration with `lsusb -d 045e:02c4`.
3. Run the standalone, no-motion Kinect bridge command above; inspect whether it says `found 1 devices` and begins publishing.
4. Only then run a safe visual integration with base/lidar drivers but **disable** task execution and movement:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch \
  start_task:=false start_move_base:=false start_wp_manager:=false navigate_enabled:=false
```

5. Validate frame rates for `/kinect2/qhd/image_color_rect` and `/kinect2/qhd/points`; then inspect YOLO debug image/detections.
6. Investigate face-verification readiness.
7. Ask for a new explicit physical-safety confirmation before enabling the actual task or navigation.

## Notes on UI

- RViz navigation configuration normally shows Map, LaserScan, robot model and waypoints. It does **not** automatically display `PointCloud2`; add a PointCloud2 display only after the point-cloud topic publishes.
- The YOLO debug viewer can start while blank; it needs image frames from Kinect.
- Do not instruct the user to click `2D Nav Goal` while verifying sensors; that sends a navigation goal.

