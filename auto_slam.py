import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import math


class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/bot5/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/bot5/goal_pose', 10)

        self.current_goal = None
        self.map_data = None
        self.robot_position = (0.0, 0.0)
        self.distance_threshold = 0.3  # 목표지점 반경 30cm 이내
        self.min_distance_from_wall = 1.0  # 벽과의 최소 거리 (미터)
        self.goal_timeout = 5.0  # 목표지점 도달 제한 시간 (초)
        self.last_goal_time = self.get_clock().now()

        self.timer = self.create_timer(1.0, self.check_goal_status)
        self.stuck_position = None
        self.stuck_check_time = self.get_clock().now()

    def map_callback(self, msg):
        self.map_data = msg

    def odom_callback(self, msg):
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def check_goal_status(self):
        if self.current_goal is None or self.is_near_current_goal() or self.is_goal_timed_out():
            self.set_new_goal()

        if self.is_stuck():
            self.rotate_180()

    def is_near_current_goal(self):
        if self.current_goal is None:
            return False

        robot_x, robot_y = self.robot_position
        goal_x, goal_y = self.current_goal
        distance = math.sqrt((robot_x - goal_x)**2 + (robot_y - goal_y)**2)
        return distance < self.distance_threshold

    def is_goal_timed_out(self):
        now = self.get_clock().now()
        return (now - self.last_goal_time).nanoseconds / 1e9 > self.goal_timeout

    def is_stuck(self):
        now = self.get_clock().now()
        if self.stuck_position is None:
            self.stuck_position = self.robot_position
            self.stuck_check_time = now
            return False

        elapsed_time = (now - self.stuck_check_time).nanoseconds / 1e9
        distance_moved = math.sqrt(
            (self.robot_position[0] - self.stuck_position[0])**2 +
            (self.robot_position[1] - self.stuck_position[1])**2
        )

        if elapsed_time > 10.0 and distance_moved < 0.1:  # 10초간 10cm 미만 이동
            return True
        elif elapsed_time > 10.0:
            self.stuck_position = self.robot_position
            self.stuck_check_time = now

        return False

    def rotate_180(self):
        self.get_logger().info("Robot is stuck. Rotating 180 degrees.")
        # 회전 동작 추가 (구체적인 회전 명령은 로봇 시스템에 맞게 구현 필요)

    def set_new_goal(self):
        new_goal = self.find_negative_goal()
        if new_goal:
            self.current_goal = new_goal
            self.last_goal_time = self.get_clock().now()
            self.send_goal(new_goal)
        else:
            self.get_logger().info("No valid goal found.")
            self.current_goal = None

    def find_negative_goal(self):
        if self.map_data is None:
            self.get_logger().info("Map data not received yet.")
            return None

        map_array = np.array(self.map_data.data).reshape(
            self.map_data.info.height, self.map_data.info.width
        )

        negative_cells = np.argwhere(map_array == -1)

        if not len(negative_cells):
            return None

        for neg_cell in negative_cells:
            neg_x, neg_y = neg_cell
            if self.is_valid_goal(neg_x, neg_y, map_array):
                wx = neg_x * self.map_data.info.resolution + self.map_data.info.origin.position.x
                wy = neg_y * self.map_data.info.resolution + self.map_data.info.origin.position.y
                return (wx, wy)

        return None

    def is_valid_goal(self, x, y, map_array):
        height, width = map_array.shape
        if 0 <= x < width and 0 <= y < height:
            return map_array[int(y), int(x)] == -1
        return False

    def send_goal(self, goal):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal[0]
        pose.pose.position.y = goal[1]
        pose.pose.orientation.w = 1.0
        self.goal_pub.publish(pose)
        self.get_logger().info(f"Sent goal: ({goal[0]}, {goal[1]})")


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
