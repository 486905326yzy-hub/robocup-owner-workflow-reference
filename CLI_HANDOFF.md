# WPB Owner Search — CLI Handoff

## Objective

Run the real WPB robot's **owner-search** demonstration as a single Docker
container: initialize ROS, base, RPLIDAR, Kinect v2, map/localization,
navigation, YOLO person detection, RViz, offline Chinese speech, owner face
verification, action recognition, and (when waving is detected) a cautious
approach.

## Safety — mandatory

This is a real mobile robot. Do **not** start a task that can navigate, rotate,
approach a person, publish `/cmd_vel`, or control the arm unless the user has
just explicitly confirmed all of these:

- The area and planned route are clear.
- An emergency stop / power-off method is within reach.
- The user is supervising the robot throughout the run.
- No person or object is within the arm's movement range.

Read-only inspection, Docker image builds, USB checks, and no-motion sensor
checks are fine without that confirmation.

## Current state

- Host: Ubuntu 24.04, NVIDIA RTX 5060, driver 595.84.
- Docker and NVIDIA container GPU access are working.
- Project: `/home/yanzy/wpb_owner_search`.
- Current image tag: `wpb-owner-search:noetic`.
- At the time of handoff, all WPB containers were intentionally stopped.
- Do **not** start `wpb-owner-search-manual`, `wpb-voice`, and a full launch at
  the same time. They publish the same ROS node names and cause each other to
  shut down.

## Stable hardware mappings

Host udev rules provide stable device names:

| Hardware | Host device | Notes |
|---|---|---|
| WPB base controller | `/dev/ftdi` | FTDI `0403:6001`, serial `AB0LPLOE` |
| RPLIDAR A2M8 | `/dev/rplidar` | CP2102 `10c4:ea60`, serial `0001`, 115200 baud |
| Kinect v2 | dynamic `/dev/bus/usb/BBB/DDD` | USB ID `045e:02c4`; determine current bus/device immediately before launch |
| Analog speaker | `/dev/snd` | ALSA `plughw:1,0` |

## Verified so far

- Base core communicates and publishes `/odom`, `/imu/data_raw`,
  `/joint_states`.
- RPLIDAR A2M8 reports healthy and scans at 115200 baud.
- Kinect v2 is detected and publishes color images and QHD RGB point clouds at
  about 15 Hz when its exact current USB character device is passed to Docker.
- YOLO World person detector runs on `cuda:0` and publishes detections/debug
  images.
- RViz can display the map, laser scan, robot model and waypoints.
- InsightFace owner verification cache and nine reference photos are present;
  offline verification initialized successfully.
- Offline Chinese TTS works through the physical speaker after setting
  `sound_play` to `plughw:1,0`.
- GPU Python 3.9 environment imports ROS bindings, custom messages, PyTorch
  2.7.1+cu128, and Ultralytics. The final live end-to-end task run remains to
  be verified under supervision.

## Fixes already applied

1. `offline_voice_bridge` launch files now default `sound_play` to
   `plughw:1,0`, the working analog speaker inside Docker.
2. ROS Noetic uses Python 3.8, but CUDA PyTorch for this RTX 50-series machine
   is installed under Python 3.9. A small relay script now replaces itself with
   Python 3.9 before the owner-search task imports ROS/Ultralytics:
   - `catkin_ws/src/wpb_task1_owner_search/scripts/task1_find_owner_real_py39.py`
   - registered in `CMakeLists.txt`
   - `task1_owner_search_real.launch` now launches this relay.
3. `Dockerfile.voice-patch` makes a small derived image and regenerates Catkin
   wrappers. It avoids re-downloading the large CUDA/PyTorch layers in the
   main `Dockerfile`.

## The only normal full-run command

Run this from a **host terminal**, not inside a container. It is a full run
and can move the robot, so only use after fresh safety confirmation.

```bash
KINECT_NODE=$(lsusb -d 045e:02c4 | awk '{gsub(":","",$2); gsub(":","",$4); printf "/dev/bus/usb/%s/%s\\n",$2,$4}')
test -e "$KINECT_NODE" || { echo "Kinect v2 not found"; exit 1; }
xhost +si:localuser:root

sudo docker run --rm -it --name wpb-owner-search-full \
  --network host --ipc host --user root --gpus all \
  -e ROS_MASTER_URI=http://127.0.0.1:11311 \
  -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device=/dev/ftdi:/dev/ftdi \
  --device=/dev/rplidar:/dev/rplidar \
  --device="$KINECT_NODE":"$KINECT_NODE" \
  --device=/dev/snd:/dev/snd \
  wpb-owner-search:noetic \
  bash -lc 'source /opt/ros/noetic/setup.bash; source /home/ubuntu20/catkin_ws/devel/setup.bash; roslaunch wpb_task1_owner_search task1_owner_search_real.launch waypoint_file:=/home/ubuntu20/waypoints.xml start_asr:=false'
```

`start_asr:=false` is deliberate: TTS is verified, microphone capture has not
yet been validated. The command starts RViz and the YOLO debug window by
default.

## Expected full task sequence

1. Confirm Kinect, lidar and odometry topics arrive (up to 90 seconds).
2. Speak: “我开始前往客厅寻找主人。”
3. Navigate to waypoint `living_room` from `/home/ubuntu20/waypoints.xml`.
4. Scan about 180° for YOLO person detections.
5. Verify a candidate against owner reference photographs with InsightFace.
6. Center the verified owner in camera view.
7. Sample about seven seconds of pose data and classify action (waving,
   falling, lying, sitting, standing, or unknown).
8. Speak the result. For a waving owner, approach to roughly 0.7 m using
   Kinect point cloud distance with a lidar forward guard, then ask whether
   help is needed.

## Stop / recovery

- Normal stop: `Ctrl+C` in the full-run terminal.
- If behavior is unexpected: use the physical emergency stop immediately.
- If Docker reports a container name conflict, inspect first:

  ```bash
  sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  ```

  Stop only the named conflicting WPB container after confirming it is the
  intended target. Never run an old `manual`/`voice` container alongside the
  single full container.

## Useful safe diagnostics

After a full launch is running, these are read-only and do not command motion:

```bash
sudo docker exec -it wpb-owner-search-full bash -lc '
  source /opt/ros/noetic/setup.bash
  source /home/ubuntu20/catkin_ws/devel/setup.bash
  rosnode list
  rostopic list | grep -E "kinect2|yolo|perception|scan|odom|voice|move_base"
'
```

If `/kinect2/qhd/image_color_rect`, `/scan`, or `/odom` publish no messages,
the task deliberately remains in its hardware-readiness stage and should not
move. Inspect container logs before restarting anything.

