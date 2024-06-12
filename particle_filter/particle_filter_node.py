# MIT License

# Copyright (c) 2024 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
ROS2 Node of Particle Filter (MCL) with 2D Laserscan in JAx
"""

# ros2 python
import rclpy
from rclpy.node import Node

# TF
from tf2_ros import TransformBroadcaster
import tf_transformations

# messages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    Pose,
    PoseArray,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)

# services
from nav_msgs.srv import GetMap

# jax, numpy, and scipy
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from scipy.ndimage import distance_transform_edt

# jax PF
from jax_pf.mcl import (
    compute_sensor_model,
    motion_update,
    sensor_update,
    mcl_init,
    mcl_init_with_pose,
    mcl_update,
)


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter")

        # declare pararmeters
        self.declare_parameter("seed")
        self.declare_parameter("lwb")
        self.declare_parameter("scan_topic")
        self.declare_parameter("odometry_topic")
        self.declare_parameter("update_on_scan")
        self.declare_parameter("angle_step")
        self.declare_parameter("num_particles")
        self.declare_parameter("theta_discretization")
        self.declare_parameter("eps")
        self.declare_parameter("max_range")
        self.declare_parameter("z_short")
        self.declare_parameter("z_max")
        self.declare_parameter("z_rand")
        self.declare_parameter("z_hit")
        self.declare_parameter("sigma_hit")
        self.declare_parameter("lambda_short")
        self.declare_parameter("motion_dispersion_x")
        self.declare_parameter("motion_dispersion_y")
        self.declare_parameter("motion_dispersion_theta")
        # get parameters
        self.seed = self.get_parameter("seed").value
        self.lwb = self.get_parameter("lwb").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.odometry_topic = self.get_parameter("odometry_topic").value
        self.update_on_scan = self.get_parameter("update_on_scan").value
        self.angle_step = self.get_parameter("angle_step").value
        self.num_particles = self.get_parameter("num_particles").value
        self.theta_discretization = self.get_parameter("theta_discretization").value
        self.eps = self.get_parameter("eps").value
        self.max_range = self.get_parameter("max_range").value
        self.z_short = self.get_parameter("z_short").value
        self.z_max = self.get_parameter("z_max").value
        self.z_rand = self.get_parameter("z_rand").value
        self.z_hit = self.get_parameter("z_hit").value
        self.sigma_hit = self.get_parameter("sigma_hit").value
        self.lambda_short = self.get_parameter("lambda_short").value
        self.motion_dispersion_x = self.get_parameter("motion_dispersion_x").value
        self.motion_dispersion_y = self.get_parameter("motion_dispersion_y").value
        self.motion_dispersion_theta = self.get_parameter(
            "motion_dispersion_theta"
        ).value
        # calculated parameters
        self.max_range_px = None
        self.theta_index_increment = None
        self.sines = None
        self.cosines = None
        self.num_beams = None
        # rng
        self.rng = jax.random.PRNGKey(self.seed)
        # data containers
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False
        self.orig_x = None
        self.orig_y = None
        self.orig_t = None
        self.orig_c = None
        self.orig_s = None
        self.height = None
        self.wigth = None
        self.resolution = None
        self.dt = None
        self.sensor_model_table = None
        self.particles = None
        self.weights = None
        self.downsampled_scan = None
        self.downsampled_theta = None
        self.last_pose = None

        # get occupancy map
        self.map_client = self.create_client(GetMap, "/map_server/map")
        # get distance transform
        self.get_omap()
        # precompute sensor model
        self.precompute_sensor_model()
        # initialize particle states
        self.initialize_particles()

        # pubs
        self.pose_pub = self.create_publisher(PoseStamped, "/pf/viz/inferred_pose", 1)
        self.particles_pub = self.create_publisher(PoseArray, "/pf/viz/particles", 1)
        self.fake_scan_pub = self.create_publisher(LaserScan, "/pf/viz/fake_scan", 1)
        self.tf_pub = TransformBroadcaster(self)

        # subs
        self.lidar_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.lidar_callback, 1
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odometry_topic, self.odom_callback, 1
        )
        self.clicked_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.clicked_pose_callback, 1
        )

        self.get_logger().info("Finished initialization, waiting on messages...")

    def lidar_callback(self, msg: LaserScan):
        if self.theta_index_increment is None:
            # first call
            self.get_logger().info("Received first LaserScan message...")
            scan = msg.ranges
            self.downsampled_scan = scan[::self.angle_step]
            theta_min = msg.angle_min
            theta_max = msg.angle_max
            angle_increment = msg.angle_increment
            theta_scan = jnp.linspace(theta_min, theta_max, num=len(scan))
            self.downsampled_theta = theta_scan[::self.angle_step]

            self.theta_index_increment = (
                self.theta_discretization * angle_increment / (2 * jnp.pi)
            )
            theta_arr = jnp.linspace(0.0, 2*jnp.pi, num=self.theta_discretization)
            self.sines = jnp.sin(theta_arr)
            self.cosines = jnp.cos(theta_arr)
            self.num_beams = len(self.downsampled_scan)
        else:
            self.downsampled_scan = msg.ranges[::self.angle_step]

        if self.update_on_scan:
            self.mcl_update()


    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        theta = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        pose = jnp.array([msg.pose.pose.position.x, msg.pose.pose.position.y, theta])
        
        if self.last_pose is None:
            # first call
            self.get_logger().info("Received first Odometry message...")
            self.last_pose = pose
        else:
            # calculate changes in states
            rot = tf_transformations.rotation_matrix(-self.last_pose[2], [0, 0, 1])
            delta = jnp.array([pose[:2] - self.last_pose[:2]])[:, None].T
            local_delta = jnp.dot(rot, delta)
            # TODO: check shape
            # TODO: set current action
            self.action = jnp.array([local_delta[0], local_delta[1], theta - self.last_pose[2]])
            self.last_pose = pose
            self.odom_initialized = True

        if not self.update_on_scan:
            self.mcl_update()

    def clicked_pose_callback(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        q = p.orientation
        theta = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        pose = jnp.array([p.position.x, p.position.y, theta])
        self.initialize_particles_with_pose(pose)

    def publish_pose_estimate(self, current_estimate: Array):
        pass

    def publish_tf(self, pose: Array):
        stamp = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "laser"
        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, pose[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_pub.sendTransform(t)

    def publish_particles(self):
        stamp = self.get_clock().now().to_msg()
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = "map"
        pose_list = []
        for i in range(self.particles.shape[0]):
            p = Pose()
            p.position.x = self.particles[i, 0]
            p.position.y = self.particles[i, 1]
            # TODO

    def publish_fake_scan(self):
        pass

    def get_omap(self):
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Get map service not available, waiting...")
        req = GetMap.Request()
        future = self.map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        map_msg = future.result().map
        map_info = map_msg.info
        self.height = map_info.height
        self.width = map_info.width
        self.resolution = map_info.resolution
        self.max_range_px = self.max_range / self.resolution
        self.orig_x = map_info.origin.position.x
        self.orig_y = map_info.origin.position.y
        q = map_info.origin.position.orientation
        self.orig_t = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.orig_c = np.cos(self.orig_t)
        self.orig_s = np.sin(self.orig_t)
        omap = np.array(map_msg.data).reshape((self.height, self.width))
        # TODO: might need to flip here?
        self.dt = self.resolution * distance_transform_edt(omap)

        self.map_initialized = True

    def precompute_sensor_model(self):
        self.sensor_model_table = compute_sensor_model(
            self.z_short,
            self.z_max,
            self.z_rand,
            self.z_hit,
            self.sigma_hit,
            self.lambda_short,
            self.max_range_px,
        )

    def motion_update(self, particles: Array, action: Array):
        pass

    def sensor_update(self, particles: Array, observation: Array):
        pass

    def initialize_particles(self):
        self.particles, self.weights, self.rng = mcl_init(
            self.rng,
            self.dt,
            self.num_particles,
            self.orig_x,
            self.orig_y,
            self.orig_c,
            self.orig_s,
            self.orig_t,
            self.resolution,
        )

    def initialize_particles_with_pose(self, pose: Array):
        self.particles, self.weights, self.rng = mcl_init(
            self.rng, pose, self.num_particles
        )

    def mcl_update(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)


if __name__ == "__main__":
    main()
