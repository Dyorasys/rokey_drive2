import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

class CleaningRobot(Node):
    def __init__(self):
        super().__init__('cleaning_robot')

        # 맵 데이터 구독
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # 네비게이션 액션 클라이언트
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # 로봇 상태 관리
        self.map_data = None 
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.05  # 기본 해상도 (m/cell)
        self.map_origin = [0, 0]
        self.robot_position = None  # 현재 위치 (x, y)
        self.robot_direction = 0  # 초기 방향 (북: 0, 동: 1, 남: 2, 서: 3)

        # 청소한 영역 관리
        self.cleaned_map = None  # 청소한 영역
        self.move_queue = []  # 이동할 좌표 저장

        # 방향 정의 (북, 동, 남, 서)
        self.dx = [-1, 0, 1, 0]
        self.dy = [0, 1, 0, -1]

        self.get_logger().info("Waiting for navigation server...")
        self.nav_client.wait_for_server()
        self.get_logger().info("Cleaning Robot Initialized")

    def map_callback(self, msg):
        """ 맵 데이터를 받아서 업데이트 """
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]

        self.map_data = np.array(msg.data, dtype=np.int8).reshape(self.map_height, self.map_width)
        '''
        if self.cleaned_map is None or self.cleaned_map.shape != self.map_data.shape:
            self.cleaned_map = np.zeros_like(self.map_data, dtype=np.uint8)  # 0: 미청소, 1: 청소 완료
        '''
        # 초기 로봇 위치 설정
        if self.robot_position is None:
            self.robot_position = self.find_start_position()

            if self.robot_position:
                x, y = self.robot_position
                #self.cleaned_map[x, y] = 1  # 시작 위치 청소
                self.get_logger().info(f"Starting at: {self.robot_position}")
                #self.cleaning_routine()  # 청소 시작
                self.mapping_routine()  # 매핑 시작

    def find_start_position(self):
        """ 맵에서 로봇의 초기 위치 찾기 (빈 공간 중간 값 사용) """
        #free_cells = np.argwhere(self.map_data == 0)  # 0인 곳만 찾기
        self.free_cells = np.argwhere(self.map_data == 0)  # 0인 곳만 찾기
        if len(self.free_cells) > 0:
            return tuple(self.free_cells[len(self.free_cells) // 2])  # 중앙 위치 선택
        return None
    
    def mapping_routine(self):
        """ 로봇이 왼쪽 방향부터 청소하며 이동하는 루틴 """
        #while True:
        #x, y = self.free_cells[0][0], self.free_cells[1][0]
        self.robot_position # 현재 로봇 위치 튜플

        # x, y = self.robot_position

        # 4방향 탐색
        moved = False
        for i in range(4):
            new_dir = (self.robot_direction + 3 - i) % 4  # 왼쪽부터 확인
            nx, ny = x + self.dx[new_dir], y + self.dy[new_dir]

            # 범위 체크 & 이동 가능 여부 확인
            if 0 <= nx < self.map_height and 0 <= ny < self.map_width:
                #if self.map_data[nx, ny] == 0 and self.cleaned_map[nx, ny] == 0:
                # 미개발 구역이 있으면
                if self.map_data[nx, ny] == -1:
                    self.robot_direction = new_dir  # 방향 변경
                    #self.move_queue.append((nx, ny))
                    moved = True
                    break  # 이동할 곳 찾았으면 종료

        if moved:
                # 이동 시작
                #next_x, next_y = self.move_queue.pop(0)
                #self.cleaned_map[next_x, next_y] = 1  # 청소 완료
                #self.robot_position = (nx, ny)
            self.get_logger().info(f"Moving to {self.robot_position}")
            self.move_to_target(nx, ny)

        else:
            # 이동할 곳이 없으면 후진
            back_x, back_y = x + self.dx[(self.robot_direction + 2) % 4], y + self.dy[(self.robot_direction + 2) % 4]
            if 0 <= back_x < self.map_height and 0 <= back_y < self.map_width and self.map_data[back_x, back_y] == 0:
                self.robot_position = (back_x, back_y)
                self.get_logger().info(f"Moving Backward to {self.robot_position}")
                self.move_to_target(back_x, back_y)
            else:
                self.get_logger().info("Cleaning Complete!")
                #break  # 더 이상 이동할 곳이 없으면 종료
    
    '''
    def cleaning_routine(self):
        """ 로봇이 왼쪽 방향부터 청소하며 이동하는 루틴 """
        while True:
            x, y = self.robot_position

            # 4방향 탐색
            moved = False
            for i in range(4):
                new_dir = (self.robot_direction + 3 - i) % 4  # 왼쪽부터 확인
                nx, ny = x + self.dx[new_dir], y + self.dy[new_dir]

                # 범위 체크 & 이동 가능 여부 확인
                if 0 <= nx < self.map_height and 0 <= ny < self.map_width:
                    if self.map_data[nx, ny] == 0 and self.cleaned_map[nx, ny] == 0:
                        self.robot_direction = new_dir  # 방향 변경
                        self.move_queue.append((nx, ny))
                        moved = True
                        break  # 이동할 곳 찾았으면 종료

            if moved:
                # 이동 시작
                next_x, next_y = self.move_queue.pop(0)
                self.cleaned_map[next_x, next_y] = 1  # 청소 완료
                self.robot_position = (next_x, next_y)
                self.get_logger().info(f"Moving to {self.robot_position}")
                self.move_to_target(next_x, next_y)

            else:
                # 이동할 곳이 없으면 후진
                back_x, back_y = x + self.dx[(self.robot_direction + 2) % 4], y + self.dy[(self.robot_direction + 2) % 4]
                if 0 <= back_x < self.map_height and 0 <= back_y < self.map_width and self.map_data[back_x, back_y] == 0:
                    self.robot_position = (back_x, back_y)
                    self.get_logger().info(f"Moving Backward to {self.robot_position}")
                    self.move_to_target(back_x, back_y)
                else:
                    self.get_logger().info("Cleaning Complete!")
                    break  # 더 이상 이동할 곳이 없으면 종료
    '''

    def move_to_target(self, x, y):
        """ 목표 위치로 이동 """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Navigation server not available!")
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x * self.map_resolution + self.map_origin[0]
        goal.pose.pose.position.y = y * self.map_resolution + self.map_origin[1]
        goal.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Sending Move Request: {goal.pose.pose.position.x}, {goal.pose.pose.position.y}")

        send_goal_future = self.nav_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """ 이동 목표 도착 여부 확인 """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by server")
            return

        self.get_logger().info("Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """ 목표 도착 후 처리 """
        result = future.result()
        if result:
            self.get_logger().info("Reached target, resuming cleaning...")
            self.cleaning_routine()
        else:
            self.get_logger().warn("Goal execution failed!")

def main(args=None):
    rclpy.init(args=args)
    robot = CleaningRobot()
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()