"""
ROS2 Node of Particle Filter (MCL) with 2D Laserscan in JAX

Author: Hongrui Zheng
Last Modified: Jun 15, 2024
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
from std_msgs.msg import Float32MultiArray
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
import os
import numpy as np
import jax
from jax import Array
from scipy.ndimage import distance_transform_edt

# jax PF
from jax_pf.mcl import (
    compute_sensor_model,
    mcl_init,
    mcl_init_with_pose,
    mcl_update,
)
from jax_pf.ray_marching import get_scan

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter")

        # declare pararmeters
        self.declare_parameter("seed", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("scan_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("odometry_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("update_on_scan", rclpy.Parameter.Type.BOOL)
        self.declare_parameter("angle_step", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("num_particles", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("theta_discretization", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("eps", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("max_range", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("z_short", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("z_max", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("z_rand", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("z_hit", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("sigma_hit", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("lambda_short", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("motion_dispersion_x", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("motion_dispersion_y", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("motion_dispersion_theta", rclpy.Parameter.Type.DOUBLE)
        # get parameters
        self.seed = self.get_parameter("seed").value
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
        self.current_estimate = None
        self.downsampled_scan = None
        self.downsampled_theta = None
        self.last_pose = None
        self.odom_updated = False
        self.action = None

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

        # timer
        self.timer = self.create_timer(0.001, self.timer_callback)

        self.get_logger().info("Finished initialization, waiting on messages...")

    def timer_callback(self):
        if self.odom_updated:
            self.mcl_update()
            self.odom_updated = False
        elif self.action is not None:
            self.action = np.zeros_like(self.action)
            self.mcl_update()

    def lidar_callback(self, msg: LaserScan):
        if self.theta_index_increment is None:
            # first call
            self.get_logger().info("Received first LaserScan message...")
            scan = msg.ranges
            self.downsampled_scan = np.clip(
                np.array(scan[:: self.angle_step]), a_min=0.0, a_max=self.max_range
            )
            self.num_beams = len(self.downsampled_scan)
            self.theta_min = msg.angle_min
            self.theta_max = msg.angle_max
            self.fov = self.theta_max - self.theta_min
            # self.angle_increment = msg.angle_increment
            self.angle_increment = self.fov / (self.num_beams - 1)
            theta_scan = np.linspace(self.theta_min, self.theta_max, num=len(scan))
            self.downsampled_theta = theta_scan[:: self.angle_step]
            self.theta_index_increment = (
                self.theta_discretization * self.angle_increment / (2 * np.pi)
            )
            theta_arr = np.linspace(0.0, 2 * np.pi, num=self.theta_discretization)
            self.sines = np.sin(theta_arr)
            self.cosines = np.cos(theta_arr)
        else:
            self.downsampled_scan = np.clip(
                np.array(msg.ranges[:: self.angle_step]),
                a_min=0.0,
                a_max=self.max_range,
            )

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        theta = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, theta])

        if self.last_pose is None:
            # first call
            self.get_logger().info("Received first Odometry message...")
            self.last_pose = pose
        else:
            # calculate changes in states
            rot = tf_transformations.rotation_matrix(-self.last_pose[2], [0, 0, 1])
            delta = np.array([pose[:2] - self.last_pose[:2]]).T
            local_delta = np.dot(rot[:2, :2], delta)
            self.action = np.array(
                [local_delta[0][0], local_delta[1][0], theta - self.last_pose[2]]
            )
            self.last_pose = pose
        self.odom_updated = True

    def clicked_pose_callback(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        q = p.orientation
        theta = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        pose = np.array([p.position.x, p.position.y, theta])
        self.initialize_particles_with_pose(pose)

    def publish_pose_estimate(self):
        stamp = self.get_clock().now().to_msg()
        p = PoseStamped()
        p.header.stamp = stamp
        p.header.frame_id = "map"
        p.pose.position.x = float(self.current_estimate[0])
        p.pose.position.y = float(self.current_estimate[1])
        q = tf_transformations.quaternion_about_axis(
            self.current_estimate[2], [0, 0, 1]
        )
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]
        self.pose_pub.publish(p)

    def publish_tf(self):
        stamp = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "laser"
        t.transform.translation.x = float(self.current_estimate[0])
        t.transform.translation.y = float(self.current_estimate[1])
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(
            0.0, 0.0, float(self.current_estimate[2])
        )
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
            p.position.x = float(self.particles[i, 0])
            p.position.y = float(self.particles[i, 1])
            q = tf_transformations.quaternion_about_axis(
                float(self.particles[i, 2]), [0, 0, 1]
            )
            p.orientation.x = q[0]
            p.orientation.y = q[1]
            p.orientation.z = q[2]
            p.orientation.w = q[3]
            pose_list.append(p)
        pa.poses = pose_list
        self.particles_pub.publish(pa)

    def publish_fake_scan(self):
        fake_scan = get_scan(
            self.current_estimate,
            self.theta_discretization,
            self.fov,
            self.num_beams,
            self.theta_index_increment,
            self.sines,
            self.cosines,
            self.eps,
            self.orig_x,
            self.orig_y,
            self.orig_c,
            self.orig_s,
            self.height,
            self.width,
            self.resolution,
            self.dt,
            self.max_range,
        )
        ls = LaserScan()
        ls.header.stamp = self.get_clock().now().to_msg()
        ls.header.frame_id = "laser"
        ls.range_max = self.max_range + 0.1
        ls.range_min = 0.0
        ls.angle_min = self.theta_min
        ls.angle_max = self.theta_max
        ls.angle_increment = self.angle_increment
        ls.ranges = list(np.array(fake_scan, dtype=float))
        self.fake_scan_pub.publish(ls)

    def get_omap(self):
        while not self.map_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Get map service not available, waiting...")
        req = GetMap.Request()
        future = self.map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        map_msg = future.result().map
        map_info = map_msg.info
        self.height = map_info.height
        self.width = map_info.width
        self.resolution = map_info.resolution
        self.max_range_px = int(self.max_range / self.resolution)
        self.orig_x = map_info.origin.position.x
        self.orig_y = map_info.origin.position.y
        q = map_info.origin.orientation
        self.orig_t = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.orig_c = np.cos(self.orig_t)
        self.orig_s = np.sin(self.orig_t)
        omap = np.array(map_msg.data).reshape((self.height, self.width))
        omap[omap < 10] = 255
        omap[omap >= 10] = 0
        self.dt = self.resolution * distance_transform_edt(omap)
        self.dt = jax.device_put(self.dt, jax.devices()[0])

        self.get_logger().info("Map initialized.")

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
        self.sensor_model_table = jax.device_put(
            self.sensor_model_table, jax.devices()[0]
        )

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
        self.particles, self.weights, self.rng = mcl_init_with_pose(
            self.rng, pose, self.num_particles
        )

    def mcl_update(self):
        if self.action is None:
            return
        self.get_logger().info("MCL Updating")
        # mcl updates
        self.particles, self.weights, self.current_estimate, self.rng = mcl_update(
            self.rng,
            self.particles,
            self.weights,
            self.action,
            self.downsampled_scan,
            self.motion_dispersion_x,
            self.motion_dispersion_y,
            self.motion_dispersion_theta,
            self.sensor_model_table,
            self.theta_discretization,
            self.fov,
            self.num_beams,
            self.theta_index_increment,
            self.sines,
            self.cosines,
            self.eps,
            self.orig_x,
            self.orig_y,
            self.orig_c,
            self.orig_s,
            self.height,
            self.width,
            self.resolution,
            self.dt,
            self.max_range,
        )

        # inferred pose and tf
        self.publish_pose_estimate()
        self.publish_tf()

        # visualization
        if (
            self.fake_scan_pub.get_subscription_count() > 0
            and self.current_estimate is not None
        ):
            self.publish_fake_scan()
        if (
            self.particles_pub.get_subscription_count() > 0
            and self.particles is not None
        ):
            self.publish_particles()


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)


if __name__ == "__main__":
    main()
