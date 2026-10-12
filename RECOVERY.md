# Robot workflow recovery

This repository intentionally contains source code, launch files, Docker build
recipes and small configuration only. It does **not** store runtime recordings,
registered-owner data, model weights, build products or Python caches.

## Required local artifacts

- OSNet ReID checkpoint:
  `catkin_ws/src/wpb_task1_owner_search/models/reid/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`
- YOLO World checkpoint: `catkin_ws/yolov8s-world.pt`
- Torchreid source: clone `KaiyangZhou/deep-person-reid` into
  `catkin_ws/src/wpb_task1_owner_search/third_party/deep-person-reid`.
- Optional locally cached weights: `catkin_ws/weights/` and `home_cache/`.

Keep these artifacts in the team's backup storage. After restoring them, build
the existing catkin workspace; no model should be downloaded implicitly.

## Launch profiles

- Current single-room workflow: `owner_voice_reid_test.launch`
- Optimized multi-waypoint workflow: `owner_voice_reid_optimized.launch`

The optimized workflow inherits the current launch defaults, including Kinect
calibration, local ReID checkpoint, GPU ReID, stable camera rate and ASR setup.
Its patrol waypoints are `living_room`, `kitchen`, `bedroom`, `canteen`, then
`exit`; those names must exist in the waypoint XML.
