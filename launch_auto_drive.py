import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")

    # リポジトリ内の worlds/course.world を取得（カレントディレクトリまたはホームディレクトリ対応）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    world_path = os.path.join(current_dir, "worlds", "course.world")
    if not os.path.exists(world_path):
        world_path = os.path.expanduser("~/turtlebot3_ws/src/pai_final/worlds/course.world")

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world_path}.items()
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    robot_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, "launch", "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": "true"}.items()
    )

    spawn_tb3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, "launch", "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={"x_pose": "0.0", "y_pose": "-3.1", "yaw": "0.0"}.items()
    )

    return LaunchDescription([
        gzserver,
        gzclient,
        robot_state_pub,
        spawn_tb3
    ])
