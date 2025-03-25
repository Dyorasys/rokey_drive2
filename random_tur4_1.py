#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
import math

class ZigzagNonRevisitWalker(Node):
    def __init__(self):
        super().__init__('zigzag_nonrevisit_walker')
        
        # cmd_vel 퍼블리셔
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Path 퍼블리셔 (RViz에서 경로 확인)
        self.path_pub = self.create_publisher(Path, 'robot_path', 10)
        
        # LaserScan, Odometry 구독자 생성
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        
        self.laser_scan = None
        self.current_pose = (0.0, 0.0, 0.0)
        
        # 방문한 셀 기록 (재방문 회피용)
        self.visited_cells = set()
        self.grid_resolution = 0.5  # 단위: m
        
        # Path 메시지 초기화 (RViz에서 경로 확인)
        self.path = Path()
        self.path.header.frame_id = "odom"  # RViz의 Fixed Frame과 일치
        
        # 지그재그 이동 관련 파라미터
        self.forward_speed = 0.2         # 전진 속도 (m/s)
        self.angular_amplitude = 0.5       # 기본 회전 각속도 (rad/s)
        self.zigzag_period = 4.0           # 지그재그 주기 (초)
        self.start_time = self.get_clock().now()  # 시작 시간 기록
        
        # 장애물 회피 관련 파라미터
        self.obstacle_distance_threshold = 0.4  # 장애물 임계 거리 (m)
        self.obstacle_counter = 0          # 연속 장애물 감지 횟수
        self.obstacle_threshold = 20       # (타이머 0.1초 주기 기준) 약 2초 이상
        self.forced_direction = None       # 강제 회전 방향 ("left" 또는 "right")
        
        # 타이머 설정
        self.timer = self.create_timer(0.1, self.timer_callback)  # cmd_vel 발행 타이머
        self.path_timer = self.create_timer(0.5, self.publish_path) # 주기적으로 Path 발행
        
        self.get_logger().info("Zigzag Non-Revisit Walker 노드가 시작되었습니다.")

    def scan_callback(self, msg: LaserScan):
        self.laser_scan = msg

    def odom_callback(self, msg: Odometry):
        # Odometry로부터 현재 위치 및 자세 업데이트
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Quaternion -> yaw 변환
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        self.current_pose = (x, y, theta)
        
        # 방문 셀 업데이트 (grid 해상도로 변환)
        cell = self.get_cell(x, y)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            self.get_logger().info(f"새로운 셀 방문: {cell}")
        
        # Path 메시지에 현재 위치 추가 (RViz 표시용)
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "odom"
        pose_stamped.pose = msg.pose.pose
        self.path.poses.append(pose_stamped)

    def publish_path(self):
        """주기적으로 Path 메시지를 발행"""
        if self.path.poses:
            self.path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.path)

    def get_cell(self, x: float, y: float):
        """
        x, y 좌표를 grid 해상도(self.grid_resolution)를 사용하여 셀 (정수 튜플)로 변환
        """
        cell_x = int(math.floor(x / self.grid_resolution))
        cell_y = int(math.floor(y / self.grid_resolution))
        return (cell_x+1, cell_y+1)

    def simulate_pose(self, pose, twist, T=1.0, dt=0.1):
        """
        간단한 오일러 통합을 사용하여 T 시간 후의 예상 pose (x, y, theta)를 계산  
        pose: (x, y, theta)  
        twist: geometry_msgs/Twist
        """
        x, y, theta = pose
        t = 0.0
        while t < T:
            x += twist.linear.x * dt * math.cos(theta)
            y += twist.linear.x * dt * math.sin(theta)
            theta += twist.angular.z * dt
            t += dt
        return (x, y, theta)

    def get_region_values(self, scan: LaserScan, start_angle: float, end_angle: float):
        """
        LaserScan 데이터에서 지정한 각도 범위([start_angle, end_angle])의 유효한 거리 값을 리스트로 반환
        """
        if start_angle < scan.angle_min:
            start_angle = scan.angle_min
        if end_angle > scan.angle_max:
            end_angle = scan.angle_max
        start_index = int((start_angle - scan.angle_min) / scan.angle_increment)
        end_index = int((end_angle - scan.angle_min) / scan.angle_increment)
        region = []
        for r in scan.ranges[start_index:end_index+1]:
            if not math.isinf(r) and not math.isnan(r):
                region.append(r)
        if not region:
            region = [scan.range_max]
        return region

    def timer_callback(self):
        twist = Twist()
        obstacle_active = False

        # ★ 장애물 회피 분기 (LaserScan 데이터 기반)
        if self.laser_scan is not None:
            # 전방 영역 피팅을 수정: 기존 -0.5236 ~ 0.5236 대신 -1.9600 ~ -1.1800 사용
            front_values = self.get_region_values(self.laser_scan, -1.9690, -1.1710)
            if min(front_values) < self.obstacle_distance_threshold:
                obstacle_active = True
                self.obstacle_counter += 1
                # 좌/우 영역 비교
                left_values = self.get_region_values(self.laser_scan, 0.5236, 1.5708)
                right_values = self.get_region_values(self.laser_scan, -1.5708, -0.5236)
                left_avg = sum(left_values) / len(left_values)
                right_avg = sum(right_values) / len(right_values)
                
                if self.obstacle_counter < self.obstacle_threshold:
                    # 정상 장애물 회피: 좌/우 여유 비교
                    if left_avg > right_avg:
                        twist.angular.z = self.angular_amplitude
                        self.forced_direction = 'left'
                        self.get_logger().info("장애물 감지: 좌측 회전 (정상)")
                    else:
                        twist.angular.z = -self.angular_amplitude
                        self.forced_direction = 'right'
                        self.get_logger().info("장애물 감지: 우측 회전 (정상)")
                else:
                    # 장애물이 지속되면 강제 방향으로 더 빠르게 회전
                    if self.forced_direction is None:
                        self.forced_direction = 'left' if left_avg > right_avg else 'right'
                    if self.forced_direction == 'left':
                        twist.angular.z = self.angular_amplitude * 1.5
                    else:
                        twist.angular.z = -self.angular_amplitude * 1.5
                    self.get_logger().info(f"장애물 지속 감지: 강제 {self.forced_direction} 회전 (더 빠름)")
                
                twist.linear.x = 0.0
                self.cmd_vel_pub.publish(twist)
                return  # 장애물 회피 시에는 후속 재방문 회피/지그재그 동작 생략

        # ★ 장애물이 없으면 (비장애물 상황)
        # 장애물 관련 변수 초기화
        self.obstacle_counter = 0
        self.forced_direction = None
        
        # 지그재그 이동 기본 candidate (시간에 따른 정해진 회전)
        now = self.get_clock().now()
        elapsed_sec = (now - self.start_time).nanoseconds * 1e-9
        phase = elapsed_sec % self.zigzag_period
        # 기본 candidate: 전진 + 회전 (지그재그)
        candidate_twist = Twist()
        candidate_twist.linear.x = self.forward_speed
        if phase < self.zigzag_period / 2.0:
            candidate_twist.angular.z = self.angular_amplitude
        else:
            candidate_twist.angular.z = -self.angular_amplitude

        # 재방문 회피: 후보 twist의 1초 후 예측 위치가 이미 방문한 셀인지 확인
        T_sim = 1.0  # 시뮬레이션 시간 (초)
        predicted_pose = self.simulate_pose(self.current_pose, candidate_twist, T=T_sim, dt=0.1)
        predicted_cell = self.get_cell(predicted_pose[0], predicted_pose[1])
        
        candidate_used = candidate_twist  # 기본 후보
        
        if predicted_cell in self.visited_cells:
            self.get_logger().info(f"기본 candidate predicted cell {predicted_cell} 이미 방문됨. 다른 후보 시도")
            # 후보 2: 기본 candidate의 회전 방향을 반대로 변경하여 재시도
            candidate_twist2 = Twist()
            candidate_twist2.linear.x = self.forward_speed
            candidate_twist2.angular.z = -candidate_twist.angular.z  # 반대 회전
            predicted_pose2 = self.simulate_pose(self.current_pose, candidate_twist2, T=T_sim, dt=0.1)
            predicted_cell2 = self.get_cell(predicted_pose2[0], predicted_pose2[1])
            if predicted_cell2 not in self.visited_cells:
                candidate_used = candidate_twist2
                self.get_logger().info(f"후보 2 선택: 예측 셀 {predicted_cell2} (반대 회전)")
            else:
                # 후보 3: 회전 없이 직진 (회전 각속도 0)
                candidate_twist3 = Twist()
                candidate_twist3.linear.x = self.forward_speed
                candidate_twist3.angular.z = 0.0
                predicted_pose3 = self.simulate_pose(self.current_pose, candidate_twist3, T=T_sim, dt=0.1)
                predicted_cell3 = self.get_cell(predicted_pose3[0], predicted_pose3[1])
                if predicted_cell3 not in self.visited_cells:
                    candidate_used = candidate_twist3
                    self.get_logger().info(f"후보 3 선택: 예측 셀 {predicted_cell3} (직진)")
                else:
                    self.get_logger().info(f"모든 후보 실패: 기본 candidate (예측 셀 {predicted_cell}) 사용")
        
        self.cmd_vel_pub.publish(candidate_used)

def main(args=None):
    rclpy.init(args=args)
    node = ZigzagNonRevisitWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
