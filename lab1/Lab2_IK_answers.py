from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task2_inverse_kinematics import MetaData

import numpy as np
from scipy.spatial.transform import Rotation as R
# part1_inverse_kinematics
# part1_IK_V
# part1_IK_BUG1


def part1_IK_BUG1(meta_data:MetaData, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    max_iteration = 25
    min_Res = 0.005
    epsilon = 1e-8

    # Three step for Chain IK
    # 1 find Path for IK metric
    path = meta_data.get_path_from_root_to_end()[0]

    # 2 CCD Method to apply IK
    for iterator in range(max_iteration):
        # residual check
        CCD_Flag = False
        residual = np.linalg.norm(joint_positions[path[-1]] - target_pose)
        if residual <= min_Res:
            CCD_Flag = True
            break

        # end joint is end not joint, so exclude([-1])?
        for i in reversed(path[:-1]):
            Joint_Pos = joint_positions[i]
            End_Pos = joint_positions[path[-1]]

            Joint2Target = target_pose - Joint_Pos
            normJ2T = np.linalg.norm(Joint2Target)

            Joint2End = End_Pos - Joint_Pos
            normJ2E = np.linalg.norm(Joint2End)

            if normJ2E < epsilon or normJ2T < epsilon:   # XXX
                continue

            unitJ2T = Joint2Target / normJ2T
            unitJ2E = Joint2End / normJ2E

            dot_Result = np.clip(np.dot(unitJ2T, unitJ2E), -1,1)
            if dot_Result > 1.0 - epsilon :     # Orientation boundary
                continue

            # 2-1 rotation calculate
            Delta_R = R.align_vectors(
                unitJ2T.reshape(1, 3),
                unitJ2E.reshape(1, 3)
            )[0]

            residual = np.linalg.norm(joint_positions[path[-1]] - target_pose)
            if residual <= min_Res:
                CCD_Flag = True
                break

            # 2-2 apply FK to sub-joint
            currJoint = joint_positions[i].copy()
            affected = [
                k for k in range(len(joint_positions))
                if is_descendant(k, i, meta_data.joint_parent)
            ]

            # update data for all sub-joint
            for k in affected:
                if k == i : continue
                # XXX
                new_Pos = Delta_R.apply(joint_positions[k] - currJoint)
                joint_positions[k] = currJoint + new_Pos

                new_Rotation = Delta_R * R.from_quat(joint_orientations[k])
                joint_orientations[k] = new_Rotation.as_quat()

        if CCD_Flag: break

    # 3 revert if root is not the root of model

    return joint_positions, joint_orientations


def part1_inverse_kinematics(meta_data:MetaData, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    max_iteration = 25
    min_Res = 0.005
    epsilon = 1e-8

    # Three step for Chain IK
    # 1 find Path for IK metric
    path = meta_data.get_path_from_root_to_end()[0]

    # 2 CCD Method to apply IK
    for iterator in range(max_iteration):
        # residual check
        CCD_Flag = False
        residual = np.linalg.norm(joint_positions[path[-1]] - target_pose)
        if residual <= min_Res:
            break

        # end joint is end not joint, so exclude([-1])?
        for i in reversed(path[:-1]):
            Joint_Pos = joint_positions[i]
            End_Pos = joint_positions[path[-1]]

            Joint2Target = target_pose - Joint_Pos
            normJ2T = np.linalg.norm(Joint2Target)

            Joint2End = End_Pos - Joint_Pos
            normJ2E = np.linalg.norm(Joint2End)

            if normJ2E < epsilon or normJ2T < epsilon:   # XXX
                continue

            unitJ2T = Joint2Target / normJ2T
            unitJ2E = Joint2End / normJ2E

            dot_Result = np.clip(np.dot(unitJ2T, unitJ2E), -1,1)
            if dot_Result > 1.0 - epsilon :     # Orientation boundary
                continue

            # 2-1 rotation calculate
            Delta_R = R.align_vectors(
                unitJ2T.reshape(1, 3),
                unitJ2E.reshape(1, 3)
            )[0]

            # 2-2 apply FK to sub-joint
            currJoint = joint_positions[i].copy()
            affected = [
                k for k in range(len(joint_positions))
                if is_descendant(k, i, meta_data.joint_parent)
            ]

            # update data for all sub-joint
            for k in affected:
                if k != i :
                    new_Pos = Delta_R.apply(joint_positions[k] - currJoint)
                    joint_positions[k] = currJoint + new_Pos

                new_Rotation = Delta_R * R.from_quat(joint_orientations[k])
                joint_orientations[k] = new_Rotation.as_quat()

            residual = np.linalg.norm(joint_positions[path[-1]] - target_pose)
            if residual <= min_Res:
                CCD_Flag = True
                break

        if CCD_Flag: break

    # 3 revert if root is not the root of model

    return joint_positions, joint_orientations


def part1_IK_V3(meta_data:MetaData, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    max_iteration = 25
    min_Res = 0.005
    epsilon = 1e-8

    # Three step for Chain IK
    # 1 find Path for IK metric
    path = meta_data.get_path_from_root_to_end()[0]

    # 2 CCD Method to apply IK
    joint_positions, is_success = Solve_CCD_IK(
        joint_positions,
        path,
        target_pose,
        min_Res,
        25
    )

    # 3 revert if root is not the root of model

    return joint_positions, joint_orientations


def part1_IK_V4(meta_data:MetaData, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    max_iteration = 50
    min_Res = 0.005
    epsilon = 1e-8

    # Three step for Chain IK
    # 1 find Path for IK metric
    path = meta_data.get_path_from_root_to_end()[0]

    # 2 CCD Method to apply IK
    end_idx = path[-1]
    root_idx = path[0]
    chain_length = sum(
        np.linalg.norm(joint_positions[child] - joint_positions[parent])
        for parent, child in zip(path[:-1], path[1:])
    )
    is_reachable = (
        np.linalg.norm(target_pose - joint_positions[root_idx])
        <= chain_length + min_Res
    )
    affected_joints = {
        i: [
            k for k in range(len(joint_positions))
            if is_descendant(k, i, meta_data.joint_parent)
        ]
        for i in path[:-1]
    }

    def apply_delta_rotation(joint_idx, delta_rotation):
        currJoint = joint_positions[joint_idx].copy()
        for k in affected_joints[joint_idx]:
            if k != joint_idx:
                new_Pos = delta_rotation.apply(joint_positions[k] - currJoint)
                joint_positions[k] = currJoint + new_Pos

            new_Rotation = delta_rotation * R.from_quat(joint_orientations[k])
            joint_orientations[k] = new_Rotation.as_quat()

    best_residual = np.linalg.norm(joint_positions[end_idx] - target_pose)
    best_positions = joint_positions.copy()
    best_orientations = joint_orientations.copy()
    stall_count = 0
    recovery_count = 0
    stall_limit = 3
    max_recovery = 3
    recovery_angle = np.deg2rad(3.0)

    for iterator in range(max_iteration):
        # residual check
        CCD_Flag = False
        round_start_residual = np.linalg.norm(
            joint_positions[end_idx] - target_pose
        )
        residual = round_start_residual
        if residual <= min_Res:
            break

        # end joint is end not joint, so exclude([-1])?
        for i in reversed(path[:-1]):
            Joint_Pos = joint_positions[i]
            End_Pos = joint_positions[end_idx]

            Joint2Target = target_pose - Joint_Pos
            normJ2T = np.linalg.norm(Joint2Target)

            Joint2End = End_Pos - Joint_Pos
            normJ2E = np.linalg.norm(Joint2End)

            if normJ2E < epsilon or normJ2T < epsilon:   # XXX
                continue

            unitJ2T = Joint2Target / normJ2T
            unitJ2E = Joint2End / normJ2E

            dot_Result = np.clip(np.dot(unitJ2T, unitJ2E), -1,1)
            cross_Result = np.cross(unitJ2E, unitJ2T)
            cross_Norm = np.linalg.norm(cross_Result)

            # 2-1 rotation calculate
            if cross_Norm < epsilon:
                if dot_Result > 0.0:
                    continue

                reference_axes = np.eye(3)
                reference_axis = reference_axes[
                    np.argmin(np.abs(reference_axes @ unitJ2E))
                ]
                rotation_axis = np.cross(unitJ2E, reference_axis)
                rotation_axis /= np.linalg.norm(rotation_axis)
                rotation_angle = np.pi
            else:
                rotation_axis = cross_Result / cross_Norm
                rotation_angle = np.arctan2(cross_Norm, dot_Result)

            Delta_R = R.from_rotvec(rotation_axis * rotation_angle)

            # 2-2 apply FK to sub-joint
            # update data for all sub-joint
            apply_delta_rotation(i, Delta_R)

            residual = np.linalg.norm(joint_positions[end_idx] - target_pose)
            if residual < best_residual:
                best_residual = residual
                best_positions = joint_positions.copy()
                best_orientations = joint_orientations.copy()

            if residual <= min_Res:
                CCD_Flag = True
                break

        if CCD_Flag: break

        round_end_residual = np.linalg.norm(
            joint_positions[end_idx] - target_pose
        )
        if round_start_residual - round_end_residual < epsilon:
            stall_count += 1
        else:
            stall_count = 0

        if stall_count >= stall_limit:
            if not is_reachable or recovery_count >= max_recovery:
                break

            if len(path) < 3:
                break

            recovery_parent = path[-3]
            recovery_joint = path[-2]
            recovery_direction = (
                joint_positions[end_idx] - joint_positions[recovery_parent]
            )
            recovery_norm = np.linalg.norm(recovery_direction)
            if recovery_norm < epsilon:
                recovery_direction = (
                    target_pose - joint_positions[recovery_parent]
                )
                recovery_norm = np.linalg.norm(recovery_direction)
                if recovery_norm < epsilon:
                    break

            recovery_direction /= recovery_norm
            reference_axes = np.eye(3)
            reference_axis = reference_axes[
                np.argmin(np.abs(reference_axes @ recovery_direction))
            ]
            recovery_axis = np.cross(recovery_direction, reference_axis)
            recovery_axis /= np.linalg.norm(recovery_axis)
            apply_delta_rotation(
                recovery_parent,
                R.from_rotvec(recovery_axis * recovery_angle)
            )
            apply_delta_rotation(
                recovery_joint,
                R.from_rotvec(recovery_axis * (-2.0 * recovery_angle))
            )
            residual = np.linalg.norm(
                joint_positions[end_idx] - target_pose
            )
            if residual < best_residual:
                best_residual = residual
                best_positions = joint_positions.copy()
                best_orientations = joint_orientations.copy()
            recovery_count += 1
            stall_count = 0

    # 3 revert if root is not the root of model

    return best_positions, best_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations


def is_descendant(k, i, joint_parent):
    current = k

    while current != -1:
        if current == i:
            return True
        current = joint_parent[current]

    return False

def Solve_CCD_IK(joint_positions, path, target_pose, min_Res, max_iter):
    # 2 CCD Method to apply IK
    for iterator in range(max_iteration):

        for i in reversed(path[:-1]):
            Joint_Pos = joint_positions[i]
            End_Pos = joint_positions[path[-1]]

            Joint2Target = target_pose - Joint_Pos
            normJ2T = np.linalg.norm(Joint2Target)

            Joint2End = End_Pos - Joint_Pos
            normJ2E = np.linalg.norm(Joint2End)

            if normJ2E < epsilon or normJ2T < epsilon:   # XXX
                continue

            unitJ2T = Joint2Target / normJ2T
            unitJ2E = Joint2End / normJ2E

            dot_Result = np.clip(np.dot(unitJ2T, unitJ2E), -1,1)
            if dot_Result > 1.0 - epsilon :     # Orientation boundary
                continue

            # 2-1 rotation calculate
            Delta_R = R.align_vectors(
                unitJ2T.reshape(1, 3),
                unitJ2E.reshape(1, 3)
            )[0]

            # 2-2 apply FK to sub-joint
            currJoint = joint_positions[i].copy()
            affected = [
                k for k in range(len(joint_positions))
                if is_descendant(k, i, meta_data.joint_parent)
            ]

            # update data for all sub-joint
            for k in affected:
                if k != i :
                    new_Pos = Delta_R.apply(joint_positions[k] - currJoint)
                    joint_positions[k] = currJoint + new_Pos

                new_Rotation = Delta_R * R.from_quat(joint_orientations[k])
                joint_orientations[k] = new_Rotation.as_quat()

            residual = np.linalg.norm(joint_positions[path[-1]] - target_pose)
            if residual <= min_Res:
                return joint_positions, True

    return joint_positions, False