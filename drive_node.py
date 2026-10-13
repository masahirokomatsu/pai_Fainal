import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
import time

# 講義のサンプルを参考にしたCNNモデル
class TurtleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 画像の特徴を抽出する畳み込み層
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # 操舵角（ハンドル操作）を計算する全結合層
        self.fc = nn.Sequential(
            nn.Linear(6656, 100), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.fc(self.flatten(self.features(x)))

# 自律走行用のノード
class AutoDriveNode(Node):
    def __init__(self):
        super().__init__('auto_drive_node')
        self.bridge = CvBridge()

        # 学習済みモデルの読み込み
        self.device = torch.device('cpu')
        self.model = TurtleNet().to(self.device)
        model_path = '/home/ubuntu/auto_drive_model.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info('モデルの読み込みに成功しました！')

        # 学習時と同じ画像の前処理（サイズ変更と正規化）
        self.transform = transforms.Compose([
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # カメラ画像の受信設定と速度指令の送信設定
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 直進スピード（一定）
        self.linear_speed = 0.18

    # カメラ画像を受け取ったときの処理
    def image_callback(self, msg):
        try:
            # ROSの画像をOpenCV/PIL形式に変換
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # AIモデルでハンドルの向きを予測
            with torch.no_grad():
                pred_angular_z = self.model(input_tensor).item()

            # ロボットに速度指令を送る
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = float(pred_angular_z)
            self.pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f'推論エラー: {e}')

def main(args=None):
    # Ctrl+Cで通信がすぐ切れて暴走するのを防ぐ設定
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = AutoDriveNode()

    try:
        # 走行ループ開始
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+Cが押されました。ロボットを停止させます...')
    finally:
        # 確実に停止させるため、速度ゼロの指示を何回か送る（安全対策）
        stop_twist = Twist()
        for _ in range(5):
            node.pub.publish(stop_twist)
            time.sleep(0.05)
        node.get_logger().info('安全に停止しました。')

        # ノードの終了処理
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()