from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task2_inverse_kinematics import MetaData

import numpy as np
from scipy.spatial.transform import Rotation as R
import part1_ik_solvers as part1_ik
import IK_Bug as IB

# part1_inverse_kinematics

def part1_IK_Origin(meta_data:MetaData, joint_positions, joint_orientations, target_pose):
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
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    model_root = meta_data.joint_name.index("RootJoint")

    # 2 CCD Method to apply IK
    for iterator in range(max_iteration):
        # residual check
        CCD_Flag = False
        residual = np.linalg.norm(joint_positions[path[-1]] - target_pose)
        if residual <= min_Res:
            break

        # end joint is end not joint, so exclude path[-1]
        # in simple case, avoid path[0] (Root of model) of CCD rotation.
        # in hard case, avoid the root.
        for i in reversed(path[1:-1]):
            # if i == model_root:continue

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
    """    
    j_Parent = meta_data.joint_parent
    TempOri = joint_orientations.copy()
    if meta_data.root_joint != "RootJoint" :
        for i in reversed(path2[1:-1]):     # XXX
            if j_Parent[i] == -1: continue
            currOri = R.from_quat(joint_orientations[i])
            parOri = R.from_quat(joint_orientations[j_Parent[i]])
            TempOri[i] = (parOri.inv() * currOri).as_quat()

        for i in reversed(path2[1:-1]):
            joint_orientations[i] = TempOri[i]
    """
    return joint_positions, joint_orientations


def Full_Body_FK(meta_data, joint_positions, joint_orientations, local_rot_overrides=None):
    local_rot = {}
    for i in range(len(joint_positions)):
        parent = meta_data.joint_parent[i]
        if parent == -1:
            local_rot[i] = R.from_quat(joint_orientations[i])
        else:
            parent_R = R.from_quat(joint_orientations[parent])
            child_R = R.from_quat(joint_orientations[i])
            local_rot[i] = parent_R.inv() * child_R

    if local_rot_overrides is not None:
        for joint, local_R in local_rot_overrides.items():
            local_rot[joint] = local_R

    new_positions = joint_positions.copy()
    new_orientations = joint_orientations.copy()

    for i in range(len(joint_positions)):
        parent = meta_data.joint_parent[i]
        if parent == -1:
            new_positions[i] = joint_positions[i]
            new_orientations[i] = joint_orientations[i]
        else:
            parent_R = R.from_quat(new_orientations[parent])
            offset = (
                meta_data.joint_initial_position[i]
                - meta_data.joint_initial_position[parent]
            )
            new_positions[i] = new_positions[parent] + parent_R.apply(offset)
            new_orientations[i] = (parent_R * local_rot[i]).as_quat()

    return new_positions, new_orientations


def part1_IK_Origin_hard(meta_data:MetaData, joint_positions, joint_orientations, target_pose):
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
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    model_root = meta_data.joint_name.index("RootJoint")

    # 2 CCD Method to apply IK
    for iterator in range(max_iteration):
        # residual check
        CCD_Flag = False
        residual = np.linalg.norm(joint_positions[path[-1]] - target_pose)
        if residual <= min_Res:
            break

        # end joint is end not joint, so exclude path[-1]
        # in simple case, avoid path[0] (Root of model) of CCD rotation.
        # in hard case, avoid the root.
        for i in reversed(path[1:-1]):
            # if i == model_root:continue

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
                k for k in path[path.index(i):]
                # k for k in range(len(joint_positions))
                # if is_descendant(k, i, meta_data.joint_parent)
            ]

            # update data for all sub-joint
            for k in affected:
                # if i == model_root: continue

                if k != i:
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
    if meta_data.root_joint != "RootJoint" :
        j_Parent = meta_data.joint_parent
        TempOri = {}
        path2_R = list(reversed(path2))
        for parent, child in zip(path2_R[:-1], path2_R[1:]):
            #if j_Parent[child] == -1: continue
            Q_child = R.from_quat(joint_orientations[child])
            Q_parent = R.from_quat(joint_orientations[parent])
            TempOri[child] = (Q_child.inv() * Q_parent).inv()

        for parent, child in zip(path2_R[:-1], path2_R[1:]):
            #if j_Parent[child] == -1: continue
            rot =  R.from_quat(joint_orientations[parent]) * TempOri[child]
            joint_orientations[child] = rot.as_quat()


    local_rot_overrides = TempOri if meta_data.root_joint != "RootJoint" else None
    joint_positions, joint_orientations = Full_Body_FK(
        meta_data,
        joint_positions,
        joint_orientations,
        local_rot_overrides,
    )

    return joint_positions, joint_orientations


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
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

    # return part1_ik.solve_ccd(
    #     meta_data, joint_positions, joint_orientations, target_pose
    # )
    # return part1_ik.solve_autograd(
    #     meta_data, joint_positions, joint_orientations, target_pose
    # )
    return part1_ik.solve_gauss_newton(
        meta_data, joint_positions, joint_orientations, target_pose
    )


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

