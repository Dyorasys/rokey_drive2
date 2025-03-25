import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from tf2_msgs.msg import TFMessage


import numpy as np

import math
 


class mapping(Node):
    def __init__(self):
        super().__init__('mapping_node')
        #QOS_ = QoSProfile(depth=10,history=QoSHistoryPolicy.KEEP_LAST,durability=QoSDurabilityPolicy.VOLATILE)
        # map data 가져오기
        
        self.pub_goal = self.create_subscription(TFMessage, '/tf', self.timer_callback, 10)
        print('f')

    def timer_callback(self, data):
        try:
            #print(data.transforms[0].transform.translation)
            x = data.transforms[0].transform.rotation.x
            y = data.transforms[0].transform.rotation.y
            z = data.transforms[0].transform.rotation.z
            w = data.transforms[0].transform.rotation.w

            #rx, py, yz = self.euler_from_quaternion(x,y,z,w)
            tx, ty, tz = data.transforms[0].transform.translation.x, data.transforms[0].transform.translation.y, data.transforms[0].transform.translation.z

            self.cal_mat_1(x,y,z,w, tx, ty, tz)
        except:
            pass

    def cal_mat_1(self,x,y,z,w, tx, ty, tz):
        mat = np.array([[1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y), tx], [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x), ty], [2*(x*z-w*y), 2*(y*z+w*x),1-2*(x**2+y**2), tz], [0, 0, 0, 1]])
        return mat
    def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
    def cal_mat(x,y,z,tx,ty,tz):
        alpha = np.array([np.cos(z) -np.sin(z), 0, 0],[np.sin(z), np.cos(z), 0, 0],[0, 0, 0, 1],[0, 0, 0, 1])
        beta = np.array([np.cos(y) ,0, np.sin(y), 0],[0, 1, 0, 0],[-np.sin(y), 0, np.cos(y), 0],[0, 0, 0, 1])
        gamma = np.array([1, 0, 0, 0],[0, np.cos(x), -np.sin(x), 0],[0, np.sin(x), np.cos(x), 0],[0, 0, 0 ,1])
        t = beta.dot(gamma)
        t = alpha.dot(t)

        t[0][3] = tx
        t[1][3] = ty
        t[2][3] = tz
        return t

    def quaternion(q,r):
        a = np.array([[q[0], -q[1], -q[2], -q[3]], [q[1], q[0], -q[3], q[2]] ,[q[2], q[3], q[0], -q[1]] ,[q[3], -q[2], q[1], q[0]]])
        b = np.array([r[0], r[1], r[2], r[3]]).T

        return a.dot(b)


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