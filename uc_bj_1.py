import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import math
import time

class ZigzagCleaning(Node):
    def __init__(self):
        super().__init__('zigzag_cleaning')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.map_subscription = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.obstacle_detected = False  # 장애물 감지 여부
        self.linear_speed = 0.2  # 직진 속도
        self.angular_speed = math.radians(90)  # 회전 속도
        self.range_max = 12.0  # 최대 감지 거리 (Lidar에서 확인한 값)
        self.threshold = 0.4  # 장애물 감지 거리 기준
        self.front_right_distance = float('inf')  # front-right 최소 거리 저장
        self.timer = self.create_timer(0.1, self.control_loop)  # 실시간 감지 및 동작 업데이트
        self.turn_right = True  # 첫 회전은 오른쪽으로 수행
        self.turn_count = 0  # 회전 횟수 저장
        self.visited_cells = set()  # 로봇이 지나간 위치 저장
        self.map_data = None  # 맵 데이터 저장

    def map_callback(self, msg):
        """ 맵 데이터를 수신하고 저장 """
        self.map_data = msg

    def is_visited(self, x, y):
        """ 해당 좌표가 방문된 곳인지 확인 """
        if self.map_data is None:
            return False
        index = y * self.map_data.info.width + x
        return index in self.visited_cells

    def mark_visited(self, x, y):
        """ 로봇이 지나간 위치를 기록 """
        if self.map_data is not None:
            index = y * self.map_data.info.width + x
            self.visited_cells.add(index)

    def lidar_callback(self, msg):
        """ Lidar 데이터를 받아 8개 섹터로 나누고 front-right 섹터의 거리 확인 """
        num_ranges = len(msg.ranges)
        sector_size = num_ranges // 8
        
        direction_names = [
            "front", "front-right", "right", "back-right",
            "back", "back-left", "left", "front-left"
        ]
        sectors = {}
        
        for i in range(8):
            start_index = i * sector_size
            end_index = num_ranges if i == 7 else (i + 1) * sector_size
            sector_ranges = msg.ranges[start_index:end_index]
            valid_ranges = [r for r in sector_ranges if not math.isinf(r) and not math.isnan(r)]
            min_distance = min(valid_ranges) if valid_ranges else float('inf')
            sectors[direction_names[i]] = min_distance
        
        output_str = "Sector distances: " + ", ".join(
            [f"{dir}: {dist:.2f}m" if not math.isinf(dist) else f"{dir}: inf" for dir, dist in sectors.items()]
        )
        self.get_logger().info(output_str)
        
        # front-right 섹터만 고려
        self.front_right_distance = sectors.get("front-right", self.range_max)
        self.obstacle_detected = self.front_right_distance <= self.threshold

    def move_robot(self, linear_x, angular_z, duration):
        """ 로봇 이동 함수 """
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.publisher_.publish(twist)
        time.sleep(duration)
        
        # 정지
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_.publish(twist)
        
        # 현재 위치를 방문한 것으로 기록
        if self.map_data:
            x = int(self.map_data.info.origin.position.x)
            y = int(self.map_data.info.origin.position.y)
            self.mark_visited(x, y)

    def control_loop(self):
        """ 장애물 감지 후 회전 및 이동 반복 """
        if self.obstacle_detected:
            self.get_logger().info("🚧 장애물 감지! 회전 수행...")
            self.move_robot(0.0, 0.0, 0.1)  # 정지
            
            if self.turn_count < 1:
                turn_angle = math.radians(-90) if self.turn_right else math.radians(90)
                self.turn_count += 1
            else:
                turn_angle = math.radians(-90) if not self.turn_right else math.radians(90)
                self.turn_count = 0
                #self.turn_right = not self.turn_right  # 다음 방향 변경
            
            angular_speed = math.radians(90)
            rotation_duration = abs(turn_angle / angular_speed)
            
            self.move_robot(0.0, angular_speed if turn_angle > 0 else -angular_speed, rotation_duration)
            self.move_robot(0.0, 0.0, 0.1)
            self.get_logger().info("🔄 45도 회전 완료!")

            self.move_robot(0.0, angular_speed if turn_angle > 0 else -angular_speed, rotation_duration)
            self.move_robot(0.0, 0.0, 0.1)
            self.get_logger().info("🔄 45도 회전 완료!")
            
            self.move_robot(self.linear_speed, 0.0, 10.0)  # 100cm 직진
            self.move_robot(0.0, 0.0, 0.1)
            self.get_logger().info("➡ 100cm 직진 완료!")
            
            self.move_robot(0.0, angular_speed if turn_angle > 0 else -angular_speed, rotation_duration)
            self.move_robot(0.0, 0.0, 0.1)
            self.get_logger().info("🔄 45도 회전 완료!")

            self.move_robot(0.0, angular_speed if turn_angle > 0 else -angular_speed, rotation_duration)
            self.move_robot(0.0, 0.0, 0.1)
            self.get_logger().info("🔄 45도 회전 완료!")

        else:
            if not self.is_visited(0, 0):  # 현재 위치가 방문되지 않았다면 이동
                self.move_robot(self.linear_speed, 0.0, 0.1)
            
            else:
                print("갈 곳이 없습니다.")

def main(args=None):
    rclpy.init(args=args)
    node = ZigzagCleaning()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
