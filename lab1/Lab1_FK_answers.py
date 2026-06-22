import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    stack = []

    # read BVH
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()

            if line.startswith('MOTION'):
                break

            if line.startswith('ROOT'):
                joint_name.append(line.split()[1])
                joint_parent.append(-1)
                joint_offset.append([0.0, 0.0, 0.0])
                stack.append(len(joint_name) - 1)

            elif line.startswith('JOINT'):
                joint_name.append(line.split()[1])
                joint_parent.append(stack[-1])
                joint_offset.append([0.0, 0.0, 0.0])
                stack.append(len(joint_name) - 1)

            elif line.startswith('End Site'):
                parent_index = stack[-1]
                joint_name.append(joint_name[parent_index] + '_end')
                joint_parent.append(parent_index)
                joint_offset.append([0.0, 0.0, 0.0])
                stack.append(len(joint_name) - 1)

            elif line.startswith('OFFSET'):
                joint_offset[stack[-1]] = [float(x) for x in line.split()[1:4]]

            elif line.startswith('}'):
                stack.pop()

    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ

    After load_motion_data(): ndarray N * M
    """
    M = len(joint_name)
    joint_positions = np.zeros((M, 3))
    joint_orientations = np.zeros((M, 4))
    joint_rotations = [None for _ in range(M)]
    frame = motion_data[frame_id]
    channel_cursor = 0

    for i in range(M):
        if joint_parent[i] == -1:
        # root

            root_translation = frame[channel_cursor:channel_cursor + 3]
            root_rotation = R.from_euler('XYZ', frame[channel_cursor + 3:channel_cursor + 6], degrees=True)
            channel_cursor += 6

            joint_positions[i] = root_translation + joint_offset[i]
                # joint_offset can be (0,0,0) for the root cases. so pos is root_translation.
            joint_rotations[i] = root_rotation
            joint_orientations[i] = root_rotation.as_quat()

        elif joint_name[i].endswith('_end'):
            parent = joint_parent[i]
            parent_rotation = joint_rotations[parent]

            joint_positions[i] = joint_positions[parent] + parent_rotation.apply(joint_offset[i])
            joint_rotations[i] = parent_rotation
            joint_orientations[i] = parent_rotation.as_quat()

        else:
            parent = joint_parent[i]
            parent_rotation = joint_rotations[parent]
            local_rotation = R.from_euler('XYZ', frame[channel_cursor:channel_cursor + 3], degrees=True)
            channel_cursor += 3

            joint_positions[i] = joint_positions[parent] + parent_rotation.apply(joint_offset[i])
            joint_rotations[i] = parent_rotation * local_rotation
            joint_orientations[i] = joint_rotations[i].as_quat()

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
