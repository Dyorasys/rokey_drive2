# 2025-02-14
# 청소
# 최종본
# 잘 안됨
# tf 적용햇는데 뭔가 아닌듯함
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
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry, Path
import math
import cv2

import matplotlib.pyplot as plt




import numpy as np

class mapping(Node):
    def __init__(self):
        super().__init__('mapping_node')
        #QOS_ = QoSProfile(depth=10,history=QoSHistoryPolicy.KEEP_LAST,durability=QoSDurabilityPolicy.VOLATILE)

        
        # map data 가져오기
        self.sub_costmap = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        
        self.pub_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
        #self.path_pub = self.create_publisher(Path, 'robot_path', 10)
        #self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        #self.path = Path()
        #self.path.header.frame_id = "map"  # RViz의 Fixed Frame과 일치
        
        

        # data of cleaned
        self.cleaned = []
        # self.new_map = []

        self.initial_position = None
        self.last_goal = None

        #self.tf_sub = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        #self.path_timer = self.create_timer(0.5, self.publish_path) # 주기적으로 Path 발행
    '''
    def odom_callback(self, data):
        self.odom_x = data.pose.pose.position.x
        self.odom_y = data.pose.pose.position.y
    '''
    '''
    def tf_callback(self, data):
        try:
            #print(data.transforms[0].transform.translation)
            x = data.transforms[0].transform.rotation.x
            y = data.transforms[0].transform.rotation.y
            z = data.transforms[0].transform.rotation.z
            w = data.transforms[0].transform.rotation.w

            #rx, py, yz = self.euler_from_quaternion(x,y,z,w)
            tx, ty, tz = data.transforms[0].transform.translation.x, data.transforms[0].transform.translation.y, data.transforms[0].transform.translation.z

            self.tf = self.cal_mat_1(x,y,z,w, tx, ty, tz)
        except:
            pass

    def cal_mat_1(self,x,y,z,w, tx, ty, tz):
        mat = np.array([[1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y), tx], [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x), ty], [2*(x*z-w*y), 2*(y*z+w*x),1-2*(x**2+y**2), tz], [0, 0, 0, 1]])
        return mat
    '''

    def map_callback(self, data):

        #self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        resolution = data.info.resolution
        #origin_x = self.x
        #origin_y = self.y
        fx = data.info.origin.position.x
        fy = data.info.origin.position.y
        # oz = data.info.origin.position.z
        # map_mat = np.array([[ox, oy, oz, 1]])
        # result = self.tf.dot(map_mat.T)
        # #result = np.linalg.inv(self.tf).dot(map_mat.T)
        

        # fx, fy = result[0][0],result[1][0] 
        if self.initial_position is None:
            self.initial_position = (fx,fy)

        # current_index_x = int(fx/resolution)
        # current_index_y = int(fy/resolution)
        # # 중복 방지
        # flag = 0
        # for i in self.cleaned:
        #     if i[0] == current_index_x and i[1] == current_index_x:
        #         flag = 1
        #         break
            
        # if flag == 0:
        #     self.cleaned.append([current_index_x,current_index_y])

        po = PoseStamped()
        
        goal_list = []
        
        #near = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
        w = data.info.width
        h = data.info.height
        map_array = np.reshape(data.data,(h, w))
        # copy map data
        # new_map = map_array[:]

        # 크기 통일 필요!!!!!! 굳이 나눌 필요 없음
        mask = int((1/resolution))
        map_scale = 0.3
        wall_scale = 0.3
        clean_scale = 0.3
        
        # 패딩
        padded = np.pad(map_array, ((int(map_scale*mask),int(map_scale*mask)),(int(map_scale*mask),int(map_scale*mask))), 'constant', constant_values=100)
        
        # 벽 크기 키우기
        wall = np.where(padded == 100)

        for i in range(len(wall[0])):
            row, column = wall[0][i], wall[1][i]

            padded[row-int(wall_scale*mask) : row+int(wall_scale*mask)+1, column-int(wall_scale*mask) : column+int(wall_scale*mask)+1] = 100

        # 청소된 구역 표시
        for i in self.cleaned:
            # mask//2 * mask//2 구역은 청소로 표시
            padded[i[0]-int(clean_scale*mask) : i[0]+int(clean_scale*mask)+1, i[1]-int(clean_scale*mask) : i[1]+int(clean_scale*mask)+1] = 1

        # plt.figure(1)
        # plt.imshow(map_array)

        # plt.show()
        # plt.close()

        # plt.figure(2)
        # plt.imshow(padded)

        # plt.show()
        # plt.close()
        # cv2.imshow('vis', padded)
        # cv2.waitKey(1)


        # 청소 안된 지점 확인
        searched = np.where(padded == 0)
        # 현재 위치에서 가장 가까운 청소 안되는 지점으로 이동하며 청소
        for k in range(len(searched[0])):
            
            nx, ny = searched[1][k], searched[0][k]
            # distance
            goal_list.append([(nx+0.5) * resolution + fx, (ny+0.5) * resolution + fy])
            #goal_list.append([(nx+0.5) * resolution - fx, (ny+0.5) * resolution - fy])


        # 현재 위치에서 가장 가까운 점
        if len(goal_list) >0:
            if self.last_goal is not None:
                ref_x, ref_y = self.last_goal  # 마지막 목표 좌표를 기준으로 탐색
            else:
                ref_x, ref_y = self.initial_position  # 초기 위치를 기준으로 선택

            # 가장 가까운 위치
            best_goal = min(goal_list, key=lambda p: (p[0], np.hypot(p[0] - ref_x, p[1] - ref_y)))
            #best_goal = min(goal_list, key=lambda p: (p[0], np.hypot(p[0] + ref_x, p[1] + ref_y)))
            # 현재 위치 인덱스
            # current_index_x = int(ref_x/resolution)
            # current_index_y = int(ref_y/resolution)
            current_index_x = int(fx/resolution)
            current_index_y = int(fy/resolution)
            po.pose.position.x = best_goal[0]
            po.pose.position.y = best_goal[1]


            po.pose.position.z = 0.0
            po.pose.orientation.w = 1.0
            po.header.frame_id = 'map'
            po.header.stamp = self.get_clock().now().to_msg()
            # 중복 방지
            flag = 0
            for i in self.cleaned:
                if i[0] == current_index_x and i[1] == current_index_y:
                    flag = 1
                    break
            
            if flag == 0:
                self.cleaned.append([current_index_x,current_index_y])

            print('------------')
            print(f'resol = {resolution}')
            print(f'map : x = {fx}, y = {fy}')
            print(f'goal : x = {best_goal[0]}, y = {best_goal[1]}')
            print(f'current index : x = {current_index_x}, y = {current_index_y}')
            print(f'goal index : x = {int((best_goal[0]-fx)/resolution+0.5)}, y = {int((best_goal[1]-fy)/resolution+0.5)}')
            print(f'len : {len(self.cleaned)}')

            self.last_goal = (best_goal[0],best_goal[1])

            self.pub_goal.publish(po)
            
        else:
            print('done')
            # 끝나면 노드 종료?
            # super().destroy_node()

        


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