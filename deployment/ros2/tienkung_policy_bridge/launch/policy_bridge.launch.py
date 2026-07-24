from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    config = (
        Path(get_package_share_directory("tienkung_policy_bridge"))
        / "config"
        / "tienkung_policy_bridge.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="tienkung_policy_bridge",
                executable="policy_bridge_node",
                name="tienkung_policy_bridge",
                output="screen",
                parameters=[str(config)],
            )
        ]
    )

