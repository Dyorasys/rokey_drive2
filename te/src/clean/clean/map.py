# 2025-02-14
# 매핑
# 최종본
# 주석 부분은 여러 방식 시도해본 내용
# 남겨둔 이유는 원인을 파악하려 여려 시도를 해보는 동안
# 계속 지웠다 작성했다 하는게 헷갈려서 주석처리만 함

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy


import numpy as np

class mapping(Node):
    def __init__(self):
        super().__init__('mapping_node')
        # map data 가져오기
        self.sub_costmap = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.sub_costmap
        
        self.pub_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)


        self.initial_position = None
        self.last_goal = None

    def map_callback(self, data):
        # 해상도. 픽셀 - 거리 변환에 필요
        resolution = data.info.resolution
        fx = data.info.origin.position.x
        fy = data.info.origin.position.y
        if self.initial_position is None:
            self.initial_position = (fx,fy)

        po = PoseStamped()
        
        # 목표점 후보
        point_list = []
        
        # 주변 값 확인
        near = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] 
        w = data.info.width
        h = data.info.height
        # h*w 형태로 변환
        a = np.reshape(data.data,(h, w))

        # -1인 지점을 탐색
        searched = np.where(a == -1)

        # 미탐색 지점 확인        
        for k in range(len(searched[0])):
            nx, ny = searched[1][k], searched[0][k]
            # 패딩 적용시 밖으로 안나가는 지점
            if 1 < nx < w-2 and 1 < ny < h-2:
                # 8방향 확인
                for i in near:
                    # 근처에 0인 지점이 있으면
                    if a[ny+i[0]][nx+i[1]] == 0:
                        # 5*5 근처에 벽이 없으면
                        if 100 not in a[ny-1:ny+2,nx-1:nx+2]:
                            # 위치
                            point_list.append([fx + (nx+0.5) * resolution, fy + (ny+0.5) * resolution])                        
                            break
                        else:
                            break
                        
        
        # 현재 위치에서 가장 가까운 점
        if len(point_list) >= 5:
            if self.last_goal is not None:
                ref_x, ref_y = self.last_goal  # 마지막 목표 좌표를 기준으로 탐색
            else:
                ref_x, ref_y = self.initial_position  # 초기 위치를 기준으로 선택
            
            # 가장 가까운 위치
            best_goal = min(point_list, key=lambda p: (p[0], np.hypot(p[0] - ref_x, p[1] - ref_y)))
            po.pose.position.x = best_goal[0]
            po.pose.position.y = best_goal[1]
            po.pose.position.z = 0.0
            po.pose.orientation.w = 1.0
            po.header.frame_id = 'map'
            po.header.stamp = self.get_clock().now().to_msg()
            print('------------')
            print(f'resol = {resolution}')
            print(f'map : x = {fx}, y = {fy}')
            print(f'goal : x = {best_goal[0]}, y = {best_goal[1]}')
            print(f'goal_index : x = {int((best_goal[0]-fx)/resolution+0.5)}, y = {int((best_goal[1]-fy)/resolution+0.5)}')
            print(f'current_index : x = {fx//resolution}, y = {fy//resolution}')

            self.last_goal = (best_goal[0],best_goal[1])

            self.pub_goal.publish(po)

        else:
            # 도착 확인
            if (x-fx)**2 + (y-fy)**2 < 1:
                print('we are home now')
                # 끝나면 노드 종료?
                #super().destroy_node()

            else:
                print('go back home')
                x,y = self.initial_position
                po.pose.position.x = x
                po.pose.position.y = y
                po.pose.position.z = 0.0
                po.pose.orientation.w = 1.0
                po.header.frame_id = 'map'
                po.header.stamp = self.get_clock().now().to_msg()
                print('------------')
                print(f'goal : x = {best_goal[0]}, y = {best_goal[1]}')

                self.pub_goal.publish(po)
            
        


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = mapping()

    try:
        rclpy.spin(image_subscriber)
    except:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()