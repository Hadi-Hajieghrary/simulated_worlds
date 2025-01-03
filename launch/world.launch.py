#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from launch_param_builder import load_xacro, load_yaml
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.substitutions import Command
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import FindExecutable
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import RegisterEventHandler
from launch.actions import SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
import numpy as np
import math 
from scipy.spatial.transform import Rotation as R


def transform_pose_to_world_frame(frame_position, frame_orientation, pose_in_frame):
    """
    Convert a pose from the 'current frame' (Frame 1) to the 'world frame' (Frame 0).

    Assumptions:
        frame_position: [x, y, z] is the origin of Frame 1 in Frame 0 coordinates.
        frame_orientation: [roll, pitch, yaw] (radians) is the orientation of Frame 1 w.r.t. Frame 0.
        pose_in_frame: dict with:
            "position": [x, y, z] in Frame 1
            "orientation": [roll, pitch, yaw] (radians) in Frame 1

    Returns:
        dict with keys "position" and "orientation", both in Frame 0 (world).
    """
    # 1) Build T_1_to_0 (the transform that takes a point from Frame 1 → Frame 0)
    T_1_to_0 = np.eye(4)
    rotation_1_to_0 = R.from_euler('xyz', frame_orientation)
    T_1_to_0[:3, :3] = rotation_1_to_0.as_matrix()  # Rotation part
    T_1_to_0[:3, 3] = frame_position               # Translation part

    # 2) Transform the position
    p_1 = np.array(pose_in_frame["position"])          # position in Frame 1
    p_1_h = np.hstack([p_1, 1.0])                      # homogeneous coordinates
    p_0_h = T_1_to_0 @ p_1_h                           # transform to Frame 0
    p_0 = p_0_h[:3]

    # 3) Transform the orientation
    orientation_1 = R.from_euler('xyz', pose_in_frame["orientation"])  # in Frame 1
    orientation_0 = rotation_1_to_0 * orientation_1                    # combined rotation in Frame 0
    rpy_0 = orientation_0.as_euler('xyz')                              # convert back to RPY

    return {
        'position': [round(p, 4) for p in p_0.tolist()],
        'orientation': [round(p, 4) for p in rpy_0.tolist()]
    }




def quaternion_to_rpy(x, y, z, w): 
    """ Convert a quaternion into roll, pitch, yaw. 
    :param x: Quaternion x value 
    :param y: Quaternion y value
    :param z: Quaternion z value 
    :param w: Quaternion w value 
    :return: Tuple of roll, pitch, yaw in radians """ 

    # Roll (x-axis rotation) 
    sinr_cosp = 2 * (w * x + y * z) 
    cosr_cosp = 1 - 2 * (x * x + y * y) 
    roll = math.atan2(sinr_cosp, cosr_cosp) 
    
    # Pitch (y-axis rotation) 
    sinp = 2 * (w * y - z * x) 
    if abs(sinp) >= 1: 
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range 
    else: 
        pitch = math.asin(sinp) 
        
    # Yaw (z-axis rotation) 
    siny_cosp = 2 * (w * z + x * y) 
    cosy_cosp = 1 - 2 * (y * y + z * z) 
    yaw = math.atan2(siny_cosp, cosy_cosp) 
    
    return round(roll, 4), round(pitch, 4), round(yaw, 4)


def generate_launch_description():

    ld = LaunchDescription()

    package_name = 'assembly_world' 
    model_path = os.path.join(
        get_package_share_directory(package_name),
        'model',
    )
    ld.add_action(SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path))
    
    # Start Gazebo with the ROS factory plugin so we can spawn entities
    start_gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )
    ld.add_action(start_gazebo)
    previous_spawn_action = None
    world_config = load_yaml(Path(get_package_share_directory(package_name))/ 'config'/ 'world_config.yaml')
    for model_name in world_config.keys():
        package_name, relative_path = world_config[model_name]['sdf_path'][len("package://"):].split('/', 1)
        package_path = get_package_share_directory(package_name)
        sdf_file_path = os.path.join(package_path, relative_path)

        frame_id = world_config[model_name]['pose']['frame_id']
        position = world_config[model_name]['pose']['position']
        orientation = world_config[model_name]['pose']['orientation']
        if frame_id != 'world':
            if frame_id in world_config.keys():
                frame_position = [
                    float(world_config[frame_id]['pose']['position']['x']),
                    float(world_config[frame_id]['pose']['position']['y']),
                    float(world_config[frame_id]['pose']['position']['z']),
                ]
                frame_orientation = [
                    float(world_config[frame_id]['pose']['orientation']['R']),
                    float(world_config[frame_id]['pose']['orientation']['P']),
                    float(world_config[frame_id]['pose']['orientation']['Y']),
                ]
                pose_in_frame = {'position': [float(position[key]) for key in position],
                                 'orientation': [float(orientation[key]) for key in orientation]}
                new_pose = transform_pose_to_world_frame(frame_position, frame_orientation, pose_in_frame)
                position['x'] = str(new_pose['position'][0])
                position['y'] = str(new_pose['position'][1])
                position['z'] = str(new_pose['position'][2])
                orientation['R'] = str(new_pose['orientation'][0])
                orientation['P'] = str(new_pose['orientation'][1])
                orientation['Y'] = str(new_pose['orientation'][2])
            else:
                raise Exception("Frame ID does not Exists")

        # Use the spawn_entity.py node provided by gazebo_ros to load your SDF model
        spawn_entity = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name='spawn_entity',
            output='screen',
            arguments=[
                '-file', sdf_file_path,
                '-entity', model_name,
                '-x', position['x'],
                '-y', position['y'],
                '-z', position['z'],
                '-R', orientation['R'],
                '-P', orientation['P'],
                '-Y', orientation['Y'],
            ]
        )
        # If this is the first model, add it directly
        if previous_spawn_action is None:
            ld.add_action(spawn_entity)
        else:
            # Otherwise, start spawning only after the previous spawn finishes
            ld.add_action(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=previous_spawn_action,
                        on_exit=[spawn_entity],
                    )
                )
            )

        # Update the "previous" spawn action for chaining
        previous_spawn_action = spawn_entity

    # Return the LaunchDescription containing Gazebo and the spawn node
    return ld
