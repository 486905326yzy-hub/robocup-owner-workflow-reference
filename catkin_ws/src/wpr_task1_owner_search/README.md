# WPR Task 1 Owner Search

Initial simulation workflow for the first competition task in the four-room WPR world.

## Owner Reference Photo

Put the owner's reference photo here:

```bash
/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/data/owner/owner.jpg
```

Use a clear front-facing or half-body photo that matches the simulated owner's visible clothes and appearance as much as possible.

## Fast Face Verification

The task node uses lightweight InsightFace verification from the owner photo. Qwen vision fallback is disabled by default; if no face is visible or the similarity is uncertain, the robot keeps scanning instead of calling the large model.

The default config does not auto-download the face model during ROS startup. If the model is not cached, the node warns and continues without large-model fallback so the robot does not block on a slow download.

Install the optional face-recognition dependency without replacing ROS OpenCV:

```bash
cd ~/catkin_ws
src/wpr_task1_owner_search/tools/install_face_recognition_deps.sh
```

The first run downloads the lightweight InsightFace `buffalo_sc` model into `/home/ubuntu20/.insightface`. If the network is slow, rerun the command before competition testing.

Check whether the owner reference photo can produce a face embedding:

```bash
cd ~/catkin_ws
python3 src/wpr_task1_owner_search/tools/face_reference_check.py
```

## Run Full Simulation

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch wpr_task1_owner_search task1_four_rooms_owner_search.launch
```

If Gazebo and navigation are already running, start only the owner-search stack:

```bash
roslaunch wpr_task1_owner_search task1_owner_search.launch
```

## GPU Check While Running

```bash
watch -n 1 nvidia-smi
```

YOLO-World is launched with `device:=cuda:0`. Lightweight face verification runs on CPU by default so the 4GB GPU remains available for real-time person detection.

## Owner Action Recognition

After InsightFace confirms the owner, the robot stops in place, says it found the owner, centers the owner in the camera, and classifies the owner's action. The current implementation uses lightweight YOLO pose on the locked owner ROI. It does not use simulation model-state hints, and it does not call Qwen or any other large vision-language model for action recognition.

Pose model path:

```bash
/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/models/pose/yolo11n-pose.pt
```

Current supported action announcements:

- owner is sitting
- owner is lying down
- owner may have suddenly fallen
- owner is waving
- action is uncertain

## Point-Cloud Owner Approach

After the action announcement, the task starts `person_3d_projector`, reads `/perception/person_detections_3d`, and sends a `move_base` goal in `base_footprint` to stand near the owner's side. The default target keeps about 0.55 m from the owner with a small side offset so the robot does not drive directly into the person.

Useful checks while running:

```bash
rostopic echo /perception/person_detections_3d
rostopic hz /kinect2/qhd/points
```

Useful action debug logs:

```text
Owner action feature summary: samples=... lie_ratio=... sit_ratio=...
Owner action verdict: label=... confidence=... samples=...
```
