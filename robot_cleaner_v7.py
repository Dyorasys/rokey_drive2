import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Qt 관련 오류 해결

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty
import numpy as np
import cv2
import math

# ------------------ 전처리 및 경로 단순화 함수 ------------------

def bresenham_line(x0, y0, x1, y1):
    """ (x0,y0)에서 (x1,y1)까지의 정수 좌표 리스트 반환 """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def is_line_free(p1, p2, map_img, origin, resolution):
    """
    p1, p2: world 좌표 (x,y)
    map_img: 안전 free 영역 이미지 (uint8, free:255, 장애물:0)
    origin: OccupancyGrid의 origin
    resolution: m/픽셀
    → p1에서 p2까지 선분상의 모든 픽셀이 free(255)인지 검사
    """
    px1 = int((p1[0] - origin.position.x) / resolution)
    py1 = int((p1[1] - origin.position.y) / resolution)
    px2 = int((p2[0] - origin.position.x) / resolution)
    py2 = int((p2[1] - origin.position.y) / resolution)
    for (px, py) in bresenham_line(px1, py1, px2, py2):
        if px < 0 or px >= map_img.shape[1] or py < 0 or py >= map_img.shape[0]:
            return False
        if map_img[py, px] < 200:
            return False
    return True

def filter_path(path, map_img, origin, resolution):
    """ 생성된 경로(path: world 좌표 리스트)에 대해 연속 waypoint 간 free 여부 검사 """
    if not path:
        return []
    filtered = [path[0]]
    for point in path[1:]:
        if is_line_free(filtered[-1], point, map_img, origin, resolution):
            filtered.append(point)
    return filtered

def perpendicular_distance(point, start, end):
    """ 점과 선분 사이의 수직 거리 계산 """
    if start == end:
        return math.hypot(point[0]-start[0], point[1]-start[1])
    num = abs((end[0]-start[0])*(start[1]-point[1]) - (start[0]-point[0])*(end[1]-start[1]))
    den = math.hypot(end[0]-start[0], end[1]-start[1])
    return num / den

def douglas_peucker(points, epsilon):
    """ Douglas-Peucker 알고리즘으로 경로 단순화 """
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    max_dist = 0.0
    index = 0
    for i in range(1, len(points)-1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            index = i
            max_dist = dist
    if max_dist > epsilon:
        rec1 = douglas_peucker(points[:index+1], epsilon)
        rec2 = douglas_peucker(points[index:], epsilon)
        return rec1[:-1] + rec2
    else:
        return [start, end]

# ------------------ 매핑 및 청소 알고리즘 ------------------

class MappingCleaningNode(Node):
    def __init__(self):
        super().__init__('mapping_cleaning_node')
        # --- Mapping 관련 ---
        self.sub_costmap = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.sub_finish = self.create_subscription(Empty, '/finish_mapping', self.finish_mapping_callback, 10)
        
        # 매핑 단계에서는 아래 제공된 코드를 그대로 사용하여 초기 위치를 설정
        self.initial_position = None  # 아래 코드에 따라 (fx, fy)
        self.last_goal = None

        self.mode = "mapping"   # "mapping" 또는 "cleaning"
        self.final_map = None   # 최종 맵 (numpy array)
        self.map_info = None    # OccupancyGrid의 info
        
        self.latest_map = None
        self.latest_map_info = None

        # --- 청소 관련 ---
        self.robot_size = 0.342  # 로봇 크기 (m)
        self.cleaning_paths = []  # 각 영역별 청소 경로 (world 좌표 리스트의 리스트)
        self.current_region_index = 0
        self.current_goal_index = 0
        self.cleaning_timer = None

        # 청소 시작 위치: 매핑 완료 후, 아래 매핑 코드처럼 초기 위치를 사용
        self.cleaning_start_position = None

        # OpenCV 시각화용
        self.vis_map = None

        # --- cmd_vel 및 odometry ---
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.current_pose = None  # (x, y, yaw)

        self.window_closed = False

    # ---------- Mapping 단계 (아래 제공된 매핑 코드 사용) ----------
    def map_callback(self, data):
        # 아래 코드는 제공된 매핑 코드와 동일하게 동작함
        resolution = data.info.resolution
        fx = data.info.origin.position.x
        fy = data.info.origin.position.y
        if self.initial_position is None:
            self.initial_position = (fx, fy)
            self.get_logger().info(f"Initial position set to: {self.initial_position}")
        w = data.info.width
        h = data.info.height
        a = np.reshape(data.data, (h, w))
        self.latest_map = a.copy()
        self.latest_map_info = data.info

        po = PoseStamped()
        point_list = []
        near = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
        searched = np.where(a == -1)
        for k in range(len(searched[0])):
            nx, ny = searched[1][k], searched[0][k]
            if 1 < nx < w-2 and 1 < ny < h-2:
                for i in near:
                    if a[ny+i[0]][nx+i[1]] == 0:
                        if 100 not in a[ny-1:ny+2, nx-1:nx+2]:
                            point_list.append([fx + (nx+0.5)*resolution, fy + (ny+0.5)*resolution])
                            break
                        else:
                            break
        if len(point_list) >= 5:
            ref_x, ref_y = self.last_goal if self.last_goal is not None else self.initial_position
            best_goal = min(point_list, key=lambda p: math.hypot(p[0]-ref_x, p[1]-ref_y))
            po.pose.position.x, po.pose.position.y = best_goal[0], best_goal[1]
            po.pose.position.z = 0.0
            po.pose.orientation.w = 1.0
            po.header.frame_id = 'map'
            po.header.stamp = self.get_clock().now().to_msg()
            self.get_logger().info(f"Mapping goal: {best_goal}")
            self.last_goal = (best_goal[0], best_goal[1])
            self.pub_goal.publish(po)
        else:
            self.get_logger().info("Mapping done (auto).")
            zero_twist = Twist()
            self.pub_cmd_vel.publish(zero_twist)
            # 청소 시작 위치를 매핑 코드의 초기 위치로 설정
            self.cleaning_start_position = self.initial_position
            po.pose.position.x, po.pose.position.y = fx, fy
            po.pose.position.z = 0.0
            po.pose.orientation.w = 1.0
            po.header.frame_id = 'map'
            po.header.stamp = self.get_clock().now().to_msg()
            self.pub_goal.publish(po)
            self.finish_mapping_procedure()

    def finish_mapping_callback(self, msg):
        if self.mode == "mapping":
            self.get_logger().info("Finish mapping signal received!")
            po = PoseStamped()
            po.header.frame_id = 'map'
            po.header.stamp = self.get_clock().now().to_msg()
            self.cleaning_start_position = self.initial_position  # 청소 시작 위치는 초기 위치로 설정
            po.pose.position.x, po.pose.position.y = self.cleaning_start_position
            po.pose.position.z = 0.0
            po.pose.orientation.w = 1.0
            self.pub_goal.publish(po)
            self.finish_mapping_procedure()

    def finish_mapping_procedure(self):
        if self.latest_map is not None and self.latest_map_info is not None:
            self.final_map = self.latest_map.copy()
            self.map_info = self.latest_map_info
        else:
            self.get_logger().error("No map available to finish mapping.")
            return
        zero_twist = Twist()
        self.pub_cmd_vel.publish(zero_twist)
        self.mode = "cleaning"
        self.plan_cleaning()

    # ---------- 청소 영역 분할 및 경로 생성 ----------
    def plan_cleaning(self):
        if self.map_info is None or self.final_map is None:
            self.get_logger().error("No map available for cleaning planning.")
            return
        h = self.map_info.height
        w = self.map_info.width
        resolution = self.map_info.resolution
        fx = self.map_info.origin.position.x
        fy = self.map_info.origin.position.y

        # 최종 맵을 이미지로 변환 (-1은 free로 처리)
        map_img = np.zeros((h, w), dtype=np.uint8)
        map_img[self.final_map == 0] = 255
        map_img[self.final_map == 100] = 0
        map_img[self.final_map == -1] = 255

        # 모폴로지 클로징 적용하여 free 영역 병합
        kernel_close = np.ones((3,3), np.uint8)
        free_space = cv2.threshold(map_img, 200, 255, cv2.THRESH_BINARY)[1]
        free_space_closed = cv2.morphologyEx(free_space, cv2.MORPH_CLOSE, kernel_close)
        # 안전 마진 10cm 적용 (erosion); 
        # (여기서는 실제 30mm 이상 장애물은 유지되도록 safe_margin 조정 가능)
        safe_margin = 0.03  # m, 30mm
        erode_kernel_size = max(1, int(safe_margin / resolution))
        erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        safe_free = cv2.erode(free_space_closed, erode_kernel, iterations=1)
        self.vis_map = cv2.cvtColor(safe_free, cv2.COLOR_GRAY2BGR)

        # 영역 분할: contour 기반
        contours, _ = cv2.findContours(safe_free, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (1.0 / resolution)**2:
                continue
            regions.append(cnt)
        self.get_logger().info(f"Found {len(regions)} cleaning regions via contour analysis.")
        if len(regions) < 1:
            self.get_logger().info("No sufficient cleaning region found. Using entire free space as one region.")
            ys, xs = np.where(safe_free == 255)
            if len(xs) == 0 or len(ys) == 0:
                self.get_logger().error("No safe free space found for cleaning!")
                return
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            regions = [np.array([[[min_x, min_y]], [[max_x, min_y]], [[max_x, max_y]], [[min_x, max_y]]], dtype=np.int32)]

        self.cleaning_paths = []
        # 최소 영역: 1m×1m → (1/resolution)^2 픽셀
        min_area_pixels = int((1.0 / resolution) ** 2)
        for cnt in regions:
            # 영역의 bounding box 및 aspect ratio 계산 (디버깅)
            xs_cnt = cnt[:,0,0]
            ys_cnt = cnt[:,0,1]
            reg_min_x, reg_max_x = xs_cnt.min(), xs_cnt.max()
            reg_min_y, reg_max_y = ys_cnt.min(), ys_cnt.max()
            width = reg_max_x - reg_min_x
            height = reg_max_y - reg_min_y
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            self.get_logger().info(f"Region: width={width}, height={height}, aspect_ratio={aspect_ratio:.2f}")
            # 영역 내 경로 생성
            if aspect_ratio > 3.0:
                if width > height:
                    y_center = (reg_min_y + reg_max_y) / 2.0
                    path = []
                    for x in np.arange(reg_min_x, reg_max_x+1, max(1, int(self.robot_size/resolution))):
                        world_x = fx + (x+0.5)*resolution
                        world_y = fy + (y_center+0.5)*resolution
                        path.append((world_x, world_y))
                else:
                    x_center = (reg_min_x + reg_max_x) / 2.0
                    path = []
                    for y in np.arange(reg_min_y, reg_max_y+1, max(1, int(self.robot_size/resolution))):
                        world_x = fx + (x_center+0.5)*resolution
                        world_y = fy + (y+0.5)*resolution
                        path.append((world_x, world_y))
            else:
                mask_region = np.zeros_like(safe_free)
                cv2.drawContours(mask_region, [cnt], -1, 255, -1)
                path = self.plan_cleaning_path(mask_region, resolution, self.map_info.origin, safe_free)
            # 안전 시작점 보정: 영역 내에서 self.cleaning_start_position과 연결 가능한 가장 가까운 후보 선택
            safe_start = None
            if path:
                for candidate in path:
                    if is_line_free(self.cleaning_start_position, candidate, safe_free, self.map_info.origin, resolution):
                        safe_start = candidate
                        break
                if safe_start is not None:
                    index = path.index(safe_start)
                    path = path[index:]
                else:
                    self.get_logger().warn("No safe candidate found in region; using first candidate.")
            if path:
                self.cleaning_paths.append(path)
        self.get_logger().info(f"Planned cleaning paths for {len(self.cleaning_paths)} regions.")

        # 미리보기: 청소 경로들을 지도 위에 오버레이
        preview_img = self.vis_map.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        for i, path in enumerate(self.cleaning_paths):
            color = colors[i % len(colors)]
            for j in range(len(path)-1):
                pt1 = (int((path[j][0]-fx)/resolution), int((path[j][1]-fy)/resolution))
                pt2 = (int((path[j+1][0]-fx)/resolution), int((path[j+1][1]-fy)/resolution))
                cv2.line(preview_img, pt1, pt2, color, 2)
        cv2.imshow("Cleaning Path Preview", preview_img)
        self.get_logger().info("Press any key in the preview window to start cleaning...")
        cv2.waitKey(0)
        cv2.destroyWindow("Cleaning Path Preview")

        if self.cleaning_paths and len(self.cleaning_paths[0]) > 0:
            self.cleaning_start_position = self.cleaning_paths[0][0]
        self.cleaning_timer = self.create_timer(0.1, self.cleaning_timer_callback)

    def plan_cleaning_path(self, region_mask, resolution, origin, full_map_img):
        """
        해당 영역(region_mask)에서 스위핑 방식으로 경로 후보 생성 후,
        Bresenham 검사와 Douglas-Peucker 단순화를 적용하여 최종 경로 반환
        """
        ys, xs = np.where(region_mask == 255)
        if len(xs) == 0 or len(ys) == 0:
            return []
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        path = []
        # scale factor 적용: 로봇 크기의 0.5배를 spacing으로 사용
        spacing = (self.robot_size * 0.5) / resolution
        if (max_x - min_x) > (max_y - min_y):
            y = min_y
            direction = 1
            while y <= max_y:
                xs_line = np.arange(min_x, max_x+1) if direction > 0 else np.arange(max_x, min_x-1, -1)
                for x in xs_line:
                    if region_mask[int(y), int(x)] == 255:
                        world_x = origin.position.x + (x+0.5)*resolution
                        world_y = origin.position.y + (y+0.5)*resolution
                        path.append((world_x, world_y))
                y += spacing
                direction *= -1
        else:
            x = min_x
            direction = 1
            while x <= max_x:
                ys_line = np.arange(min_y, max_y+1) if direction > 0 else np.arange(max_y, min_y-1, -1)
                for y in ys_line:
                    if region_mask[int(y), int(x)] == 255:
                        world_x = origin.position.x + (x+0.5)*resolution
                        world_y = origin.position.y + (y+0.5)*resolution
                        path.append((world_x, world_y))
                x += spacing
                direction *= -1
        filtered_path = filter_path(path, full_map_img, origin, resolution)
        if not filtered_path:
            filtered_path = path
        simplified_path = douglas_peucker(filtered_path, self.robot_size / 4)
        return simplified_path

    # ---------- Odometry 콜백 ----------
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny, cosy)
        self.current_pose = (x, y, yaw)

    # ---------- 청소 실행: 경로 추종 (cmd_vel 제어) ----------
    def cleaning_timer_callback(self):
        if self.window_closed:
            return
        if self.current_region_index < len(self.cleaning_paths):
            current_path = self.cleaning_paths[self.current_region_index]
            if self.current_goal_index < len(current_path):
                target = current_path[self.current_goal_index]
                if self.current_pose is None:
                    self.get_logger().info("Waiting for odometry...")
                    return
                cur_x, cur_y, cur_yaw = self.current_pose
                dx = target[0] - cur_x
                dy = target[1] - cur_y
                distance = math.hypot(dx, dy)
                if distance < 0.1:
                    self.get_logger().info(f"Reached waypoint: {target}")
                    self.current_goal_index += 1
                else:
                    desired_angle = math.atan2(dy, dx)
                    error_angle = desired_angle - cur_yaw
                    error_angle = math.atan2(math.sin(error_angle), math.cos(error_angle))
                    k_linear = 0.5
                    k_angular = 1.5
                    linear_vel = min(k_linear * distance, 0.3)
                    angular_vel = k_angular * error_angle
                    twist = Twist()
                    twist.linear.x = linear_vel
                    twist.angular.z = angular_vel
                    self.pub_cmd_vel.publish(twist)
            else:
                self.get_logger().info(f"Region {self.current_region_index} cleaning complete.")
                self.current_region_index += 1
                self.current_goal_index = 0
        else:
            self.get_logger().info("Cleaning finished. Stopping robot.")
            twist = Twist()
            self.pub_cmd_vel.publish(twist)
            if self.cleaning_timer is not None:
                self.cleaning_timer.cancel()
        vis_copy = self.vis_map.copy()
        if self.current_pose is not None:
            res = self.map_info.resolution
            fx = self.map_info.origin.position.x
            fy = self.map_info.origin.position.y
            robot_px = int((self.current_pose[0]-fx)/res)
            robot_py = int((self.current_pose[1]-fy)/res)
            cv2.circle(vis_copy, (robot_px, robot_py), int((self.robot_size/2)/res), (0,0,255), -1)
        cv2.imshow("Cleaning Progress", vis_copy)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            self.get_logger().info("Closing visualization window by user request.")
            self.window_closed = True
            cv2.destroyWindow("Cleaning Progress")

def main(args=None):
    rclpy.init(args=args)
    node = MappingCleaningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.window_closed:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
