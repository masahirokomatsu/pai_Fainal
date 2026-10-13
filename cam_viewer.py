import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# カメラがちゃんと映るか確認するためのノード
class CamCheck(Node):
    def __init__(self):
        super().__init__('cam_check')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.cb, 10)
        print("カメラ起動チェック中... (qキーで終了)")

    def cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # そのまま表示
        cv2.imshow('turtlebot_camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main():
    rclpy.init()
    n = CamCheck()
    try:
        rclpy.spin(n)
    except:
        pass
    cv2.destroyAllWindows()
    n.destroy_node()

if __name__ == '__main__':
    main()