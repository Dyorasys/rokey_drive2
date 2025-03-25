import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from scipy.ndimage import distance_transform_edt
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool
import tf2_ros

class MapExplorer(Node):
    def __init__(self):
        super().__init__('map_explorer')

        self.subscription = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        self.start_pub = self.create_publisher(
            Bool, '/start', QoSProfile(depth=10)
        )
        
        self.publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.get_logger().info("📍 미탐색 영역 분석기 시작!")
        self.obstacle_clearance = 13  # 장애물과 최소 거리 설정 (픽셀 단위)

        self.initial_position = None  # ✅ 초기 위치 저장 변수 추가
        self.last_goal = None  # ✅ 마지막 목표 좌표 저장
        self.flag = True  # 탐색 완료 여부 플래그

        # ✅ TF2 버퍼 & 리스너 추가 (현재 위치 추적용)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def get_current_position(self):
        """ 현재 로봇의 위치를 `map` 프레임 기준으로 가져옴 """
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            return transform.transform.translation.x, transform.transform.translation.y
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("⚠ 현재 위치를 가져올 수 없음 (TF 변환 오류)")
            return None, None

    def map_callback(self, msg):
        width, height = msg.info.width, msg.info.height
        resolution = msg.info.resolution
        origin_x, origin_y = msg.info.origin.position.x, msg.info.origin.position.y

        # ✅ 초기 위치 기록 (첫 map_callback 호출 시 한 번만 저장)
        if self.initial_position is None:
            self.initial_position = (origin_x, origin_y)
            self.get_logger().info(f"🚀 초기 위치 기록: x={origin_x}, y={origin_y}")

        map_array = np.array(msg.data).reshape(height, width)

        if np.count_nonzero(map_array == -1) == 0:
            self.get_logger().info("✅ 탐색 완료! 더 이상 미탐색 영역이 없습니다.")
            return

        # 장애물(>=50)과의 거리 맵 생성
        obstacle_mask = (map_array >= 50).astype(np.uint8)
        distance_map = distance_transform_edt(1 - obstacle_mask)  # 장애물과 거리 계산

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        candidate_coords = []
        unknown_indices = np.where(map_array == -1)

        for i in range(len(unknown_indices[0])):
            ux, uy = unknown_indices[1][i], unknown_indices[0][i]

            if distance_map[uy, ux] < self.obstacle_clearance:
                continue  # 장애물과 너무 가까우면 무시

            for dx, dy in directions:
                nx, ny = ux + dx, uy + dy
                if 0 <= nx < width and 0 <= ny < height and map_array[ny, nx] == 0:
                    world_x = origin_x + (nx + 0.5) * resolution
                    world_y = origin_y + (ny + 0.5) * resolution
                    candidate_coords.append((world_x, world_y))
                    break  # 유효한 좌표 발견 시 추가 후 종료

        if candidate_coords:
            # ✅ 마지막 목표 좌표 기준으로 가장 가까운 미탐색 지점 선택 (x 좌표 우선)
            if self.last_goal is not None:
                ref_x, ref_y = self.last_goal  # 마지막 목표 좌표를 기준으로 탐색
            else:
                ref_x, ref_y = self.initial_position  # 초기 위치를 기준으로 선택
            
            best_goal = min(candidate_coords, key=lambda p: (p[0], np.hypot(p[0] - ref_x, p[1] - ref_y)))
            self.send_goal(best_goal[0], best_goal[1])
        else:
            self.get_logger().info("⚠ 이동 가능한 탐색 지점이 없음! 초기 위치로 복귀합니다.")
            #self.get_logger().info(f"current_pose{self.get_current_position()}")
            self.flag = False

            if self.initial_position is not None:
                self.send_goal(self.initial_position[0] + 1.0, self.initial_position[1] + 0.4)
            else:
                self.get_logger().info("⚠ 초기 위치 복귀 실패: 초기 위치 정보 없음.")

    def check_arrival(self):
        """ 로봇이 초기 위치에 도착했는지 확인 """
        if self.initial_position is None:
            return False  # 초기 위치 정보가 없으면 체크 불가능

        current_x, current_y = self.get_current_position()
        if current_x is None or current_y is None:
            return False  # 현재 위치를 가져오지 못하면 체크 불가능

        distance = np.hypot(current_x - self.initial_position[0], current_y - self.initial_position[1])
        return distance < 0.3  # 0.1m 이내로 도착하면 True 반환

    def send_goal(self, x, y):
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0

        if self.flag:
            self.publisher.publish(goal)
            self.get_logger().info(f"🛑 목표 좌표 전송: x={x}, y={y}")
            self.last_goal = (x, y)  # ✅ 마지막 목표 좌표 저장
        else:
            if self.check_arrival():  # ✅ 초기 위치에 도착했는지 확인
                self.get_logger().info("🛑 초기 위치 복귀 완료!")
                s_msg = Bool()
                s_msg.data = True
                self.start_pub.publish(s_msg)
                #self.flag = False
            else:
                self.get_logger().info("🛑 초기 위치 복귀 중...")
                self.publisher.publish(goal)

def main(args=None):
    rclpy.init(args=args)
    node = MapExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
