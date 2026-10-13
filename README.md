# Physical AI 応用1 最終課題
手動操縦のデータを学習させてコースを周回させる自律走行

シミュレータ（Gazebo）のTurtleBot3をコントローラーで手動操縦した走行データを集めてAIに学習させ、カメラの映像だけを見てコースを周回できるようにしました。

## 構成ファイル
- launch_auto_drive.py : コースとロボットを起動するファイル
- extract_dataset.py : 自分で走らせた走行ログから、写真とハンドルの動きを取り出すファイル
- train_model.py : 写真を見てハンドル操作を予測するAIを学習させるファイル
- drive_node.py : 学習したAIを使ってロボットを自動で走らせるファイル
- cam_viewer.py : ロボットのカメラ映像を見るファイル
- worlds/course.world : 自作したコースのワールドファイル

## 動かし方
1. コースを起動する:
   ros2 launch launch_auto_drive.py
   （※launchファイル内で worlds/course.world を自動で読み込みます）
2. 自動走行をスタートする:
   python3 drive_node.py