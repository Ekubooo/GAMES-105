import numpy as np
from scipy.spatial.transform import Rotation as R
from Lab1_FK_answers import load_motion_data


def part4_general_retarget_func(target_bvh_path, source_bvh_path, joint_mapping=None):
    """
    Part4: retarget a general source BVH motion to a target BVH skeleton.

    joint_mapping is optional and uses this direction:
        {target_joint_name: source_joint_name}
    If it is not provided, joints with the same name are matched automatically.
    """

    # 1. Parse source/target BVH skeleton and channel layout.
    target_data = parse_bvh_skeleton_with_channels(target_bvh_path)
    source_data = parse_bvh_skeleton_with_channels(source_bvh_path)
    source_motion_data = load_motion_data(source_bvh_path)

    target_names, target_parents, target_offsets, target_channels, target_channel_map, target_channel_count = target_data
    source_names, source_parents, source_offsets, source_channels, source_channel_map, source_channel_count = source_data

    # 2. Build target-to-source joint mapping.
    joint_mapping = build_general_joint_mapping(target_names, target_parents, source_names, source_parents, joint_mapping)

    # 3. Compute rest-pose correction rotations.
    correction_map = compute_rest_pose_corrections(
        target_names, target_parents, target_offsets,
        source_names, source_parents, source_offsets,
        joint_mapping
    )

    # 4. Allocate target motion_data.
    retarget_motion_data = np.zeros((source_motion_data.shape[0], target_channel_count))

    # 5. Copy root channels.
    target_root = find_root_index(target_parents)
    target_root_name = target_names[target_root]
    source_root_name = joint_mapping.get(target_root_name)

    if source_root_name in source_names:
        source_root = source_names.index(source_root_name)
        source_root_range = source_channel_map[source_root_name]
        target_root_range = target_channel_map[target_root_name]

        if source_root_range is not None and target_root_range is not None:
            target_start, target_end = target_root_range
            root_translation = extract_ordered_channel_values(
                source_motion_data, source_root_range, source_channels[source_root], 'position', 'XYZ'
            )
            root_order, root_euler = extract_rotation_channels(
                source_motion_data, source_root_range, source_channels[source_root]
            )

            retarget_motion_data[:, target_start:target_start + 3] = root_translation
            if root_order:
                root_rotation = R.from_euler(root_order, root_euler, degrees=True)
                retarget_motion_data[:, target_start + 3:target_start + 6] = root_rotation.as_euler('XYZ', degrees=True)

    # 6. Retarget each matched joint rotation.
    for target_index, target_name in enumerate(target_names):
        if target_index == target_root or target_name.endswith('_end'):
            continue
        if target_name not in joint_mapping:
            continue
        if target_channel_map[target_name] is None:
            continue

        source_name = joint_mapping[target_name]
        if source_name not in source_names or source_channel_map[source_name] is None:
            continue

        source_index = source_names.index(source_name)
        source_range = source_channel_map[source_name]
        target_start, target_end = target_channel_map[target_name]
        source_order, source_euler = extract_rotation_channels(
            source_motion_data, source_range, source_channels[source_index]
        )

        if not source_order:
            continue

        correction_rotation = correction_map.get(target_name, R.identity())
        retarget_motion_data[:, target_start:target_start + 3] = apply_retarget_rotation(
            source_euler, source_order, correction_rotation
        )

    return retarget_motion_data


def parse_bvh_skeleton_with_channels(bvh_file_path):
    joint_name = []
    joint_parent = []
    joint_offset = []
    joint_channels = []
    stack = []

    with open(bvh_file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('MOTION'):
                break

            if line.startswith('ROOT'):
                joint_name.append(line.split()[1])
                joint_parent.append(-1)
                joint_offset.append([0.0, 0.0, 0.0])
                joint_channels.append([])
                stack.append(len(joint_name) - 1)

            elif line.startswith('JOINT'):
                joint_name.append(line.split()[1])
                joint_parent.append(stack[-1])
                joint_offset.append([0.0, 0.0, 0.0])
                joint_channels.append([])
                stack.append(len(joint_name) - 1)

            elif line.startswith('End Site'):
                parent_index = stack[-1]
                joint_name.append(joint_name[parent_index] + '_end')
                joint_parent.append(parent_index)
                joint_offset.append([0.0, 0.0, 0.0])
                joint_channels.append([])
                stack.append(len(joint_name) - 1)

            elif line.startswith('OFFSET'):
                joint_offset[stack[-1]] = [float(x) for x in line.split()[1:4]]

            elif line.startswith('CHANNELS'):
                words = line.split()
                channel_count = int(words[1])
                joint_channels[stack[-1]] = words[2:2 + channel_count]

            elif line.startswith('}'):
                stack.pop()

    channel_map = {}
    channel_cursor = 0
    for name, channels in zip(joint_name, joint_channels):
        if len(channels) == 0:
            channel_map[name] = None
        else:
            channel_map[name] = (channel_cursor, channel_cursor + len(channels))
            channel_cursor += len(channels)

    return joint_name, joint_parent, np.array(joint_offset), joint_channels, channel_map, channel_cursor


def build_general_joint_mapping(target_names, target_parents, source_names, source_parents, joint_mapping=None):
    mapping = {} if joint_mapping is None else dict(joint_mapping)
    source_name_set = set(source_names)

    for target_name in target_names:
        if target_name.endswith('_end'):
            continue
        if target_name not in mapping and target_name in source_name_set:
            mapping[target_name] = target_name

    target_root_name = target_names[find_root_index(target_parents)]
    source_root_name = source_names[find_root_index(source_parents)]
    if target_root_name not in mapping:
        mapping[target_root_name] = source_root_name

    return mapping


def compute_rest_pose_corrections(target_names, target_parents, target_offsets,
                                  source_names, source_parents, source_offsets,
                                  joint_mapping):
    corrections = {}
    target_positions = calculate_rest_global_positions(target_parents, target_offsets)
    source_positions = calculate_rest_global_positions(source_parents, source_offsets)
    target_children = build_children_list(target_parents)

    source_index_by_name = {name: i for i, name in enumerate(source_names)}

    for target_name, source_name in joint_mapping.items():
        if target_name not in target_names or source_name not in source_index_by_name:
            continue

        target_index = target_names.index(target_name)
        source_index = source_index_by_name[source_name]
        matched_child = None

        for child_index in target_children[target_index]:
            child_name = target_names[child_index]
            source_child_name = joint_mapping.get(child_name)
            if source_child_name in source_index_by_name:
                matched_child = child_index
                break

        if matched_child is None:
            corrections[target_name] = R.identity()
            continue

        source_child_name = joint_mapping[target_names[matched_child]]
        source_child_index = source_index_by_name[source_child_name]
        target_direction = target_positions[matched_child] - target_positions[target_index]
        source_direction = source_positions[source_child_index] - source_positions[source_index]
        corrections[target_name] = rotation_between_vectors(target_direction, source_direction)

    return corrections


def apply_retarget_rotation(source_euler, source_order, correction_rotation):
    source_euler = np.asarray(source_euler)
    if len(source_order) == 1 and source_euler.ndim == 2:
        source_euler = source_euler[:, 0]

    source_rotation = R.from_euler(source_order, source_euler, degrees=True)
    target_rotation = source_rotation * correction_rotation
    return target_rotation.as_euler('XYZ', degrees=True)


def extract_rotation_channels(motion_data, channel_range, channels):
    if channel_range is None:
        return '', np.zeros((motion_data.shape[0], 0))

    start, end = channel_range
    rotation_indices = []
    rotation_order = ''

    for local_index, channel_name in enumerate(channels):
        if channel_name.lower().endswith('rotation'):
            rotation_indices.append(start + local_index)
            rotation_order += channel_name[0].upper()

    if len(rotation_indices) == 0:
        return '', np.zeros((motion_data.shape[0], 0))

    return rotation_order, motion_data[:, rotation_indices]


def extract_ordered_channel_values(motion_data, channel_range, channels, channel_type, order):
    values = np.zeros((motion_data.shape[0], len(order)))
    start, end = channel_range

    for output_index, axis in enumerate(order):
        target_channel = axis.lower() + channel_type.lower()
        for local_index, channel_name in enumerate(channels):
            if channel_name.lower() == target_channel:
                values[:, output_index] = motion_data[:, start + local_index]
                break

    return values


def calculate_rest_global_positions(joint_parents, joint_offsets):
    positions = np.zeros_like(joint_offsets)
    for i in range(len(joint_parents)):
        parent = joint_parents[i]
        if parent == -1:
            positions[i] = joint_offsets[i]
        else:
            positions[i] = positions[parent] + joint_offsets[i]
    return positions


def build_children_list(joint_parents):
    children = [[] for _ in joint_parents]
    for index, parent in enumerate(joint_parents):
        if parent != -1:
            children[parent].append(index)
    return children


def find_root_index(joint_parents):
    for index, parent in enumerate(joint_parents):
        if parent == -1:
            return index
    raise ValueError('BVH skeleton has no root joint')


def rotation_between_vectors(from_vector, to_vector):
    from_norm = np.linalg.norm(from_vector)
    to_norm = np.linalg.norm(to_vector)

    if from_norm < 1e-8 or to_norm < 1e-8:
        return R.identity()

    from_direction = from_vector / from_norm
    to_direction = to_vector / to_norm
    dot_value = np.clip(np.dot(from_direction, to_direction), -1.0, 1.0)

    if dot_value > 1.0 - 1e-8:
        return R.identity()

    if dot_value < -1.0 + 1e-8:
        axis = np.cross(from_direction, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(from_direction, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(axis * np.pi)

    axis = np.cross(from_direction, to_direction)
    axis_norm = np.linalg.norm(axis)
    angle = np.arctan2(axis_norm, dot_value)
    axis = axis / axis_norm
    return R.from_rotvec(axis * angle)
