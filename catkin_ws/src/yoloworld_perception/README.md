# yoloworld_perception

ROS Noetic wrapper for YOLO-World person detection. The package is intentionally
separate from `wpr_simulation` and `wpb_home`, so simulation and hardware can use
the same perception topics.

## Topics

- `/perception/person_detections_2d`: `perception_msgs/Detection2DArray`
- `/perception/yoloworld/debug_image`: `sensor_msgs/Image`
- `/perception/person_detections_3d`: `perception_msgs/Person3DArray`

## GPU smoke test

```bash
cd ~/catkin_ws/src/yoloworld_perception/models
python3 ../scripts/yoloworld_smoke_test.py --model yolov8s-world-person.pt --device cuda:0
```

## Build

```bash
cd ~/catkin_ws
catkin_make --pkg perception_msgs yoloworld_perception
source devel/setup.bash
```

## Launch

Simulation:

```bash
roslaunch yoloworld_perception yoloworld_sim.launch
```

The simulation launch opens two OpenCV windows by default: raw WPR camera image
and YOLO-World debug image. Disable them with `show_viewer:=false`.

Real robot 2D detection:

```bash
roslaunch yoloworld_perception yoloworld_real.launch
```

Real robot 2D + Kinect point-cloud 3D projection:

```bash
roslaunch yoloworld_perception yoloworld_real_3d.launch
```
