# Particle Filter Localization

This code implements the ROS2 wrapper of JAX-based MCL algorithm for F1TENTH.

This ROS2 node wraps around [jax_pf](https://github.com/hzheng40/jax_pf) for fast 2D ray marching.

# Installation

ROS 2 dependencies:
```
rosdep update
rosdep install -r --from-paths src -y
```

JAX-based ray marching and MCL [jax_pf](https://github.com/hzheng40/jax_pf):

Note that you might need customized installation of JAX. And you might need to modify the pip install to install for the python that ROS2 uses.

```
git clone https://github.com/hzheng40/jax_pf
cd jax_pf
pip install -e .
```

# Usage

Parameters are in ```config/localize.yaml```. You may have to modify the "odometry_topic" or "scan_topic" parameters to match your topic names.

Launch localization with:
```
ros2 launch particle_filter localize_launch.py
```

Once the particle filter is running, you can visualize the map and other particle filter visualization message in rviz. Use the "2D Pose Estimate" tool from the rviz toolbar to initialize the particle locations.

The following topics are available for visualization:

- Current pose estimate: ```pf/viz/inferred_pose``` (```PoseStamped```)
- Current particles: ```pf/viz/particles``` (```PoseArray```)
- Simulated scan from pose estimate: ```pf/viz/fake_scan``` (```LaserScan```)