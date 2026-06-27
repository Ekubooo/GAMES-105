"""Modular solvers for Lab1 Task2 Part1 inverse kinematics."""

import numpy as np
from scipy.spatial.transform import Rotation as R

class _Part1IKContext:
    """Store solver state and rotate subtrees after rerooting the skeleton."""

    def __init__(self, meta_data, joint_positions, joint_orientations, target_pose):
        self.meta_data = meta_data
        self.positions = np.asarray(joint_positions, dtype=np.float64).copy()
        self.orientations = np.asarray(joint_orientations, dtype=np.float64).copy()
        self.target = np.asarray(target_pose, dtype=np.float64).reshape(3)
        self.path = list(meta_data.get_path_from_root_to_end()[0])
        if len(self.path) < 2:
            raise ValueError("IK path must contain at least two joints.")
        if self.positions.shape != (len(meta_data.joint_parent), 3):
            raise ValueError("joint_positions has an invalid shape.")
        if self.orientations.shape != (len(meta_data.joint_parent), 4):
            raise ValueError("joint_orientations has an invalid shape.")

        self.root, self.end = self.path[0], self.path[-1]
        self.root_position = self.positions[self.root].copy()
        self._normalise_quaternions()

        # Reroot the undirected skeleton at the joint that must remain fixed.
        self.children = self._build_rerooted_children()
        self.subtrees = {
            joint: self._collect_subtree(joint) for joint in self.path[:-1]
        }
        self.chain_length = sum(
            np.linalg.norm(self.positions[b] - self.positions[a])
            for a, b in zip(self.path[:-1], self.path[1:])
        )
        self.reachable = (
            np.linalg.norm(self.target - self.root_position)
            <= self.chain_length + 0.005
        )
        self.best_error = self.error()
        self.best_state = self.snapshot()

    def _build_rerooted_children(self):
        count = len(self.meta_data.joint_parent)
        neighbours = [[] for _ in range(count)]
        for child, parent in enumerate(self.meta_data.joint_parent):
            if parent != -1:
                neighbours[parent].append(child)
                neighbours[child].append(parent)

        children = [[] for _ in range(count)]
        visited, queue = {self.root}, [self.root]
        while queue:
            parent = queue.pop(0)
            for child in neighbours[parent]:
                if child not in visited:
                    visited.add(child)
                    children[parent].append(child)
                    queue.append(child)
        if len(visited) != count:
            raise ValueError("The skeleton must be a connected tree.")
        return children

    def _collect_subtree(self, root):
        result, stack = [], [root]
        while stack:
            joint = stack.pop()
            result.append(joint)
            stack.extend(self.children[joint])
        return np.asarray(result, dtype=np.int64)

    def _normalise_quaternions(self):
        norms = np.linalg.norm(self.orientations, axis=1, keepdims=True)
        invalid = norms[:, 0] < 1e-12
        self.orientations[~invalid] /= norms[~invalid]
        self.orientations[invalid] = np.array([0.0, 0.0, 0.0, 1.0])

    def snapshot(self):
        return self.positions.copy(), self.orientations.copy()

    def restore(self, state):
        self.positions[...] = state[0]
        self.orientations[...] = state[1]

    def error(self):
        return float(np.linalg.norm(self.positions[self.end] - self.target))

    def remember_best(self):
        current_error = self.error()
        if current_error < self.best_error:
            self.best_error = current_error
            self.best_state = self.snapshot()

    def apply_world_rotation(self, joint, delta_rotation):
        """Rotate a rerooted subtree around joint and update global quaternions."""
        affected = self.subtrees[joint]
        pivot = self.positions[joint].copy()
        matrix = delta_rotation.as_matrix()
        self.positions[affected] = (
            (self.positions[affected] - pivot) @ matrix.T + pivot
        )
        current = R.from_quat(self.orientations[affected])
        self.orientations[affected] = (delta_rotation * current).as_quat()
        self.positions[self.root] = self.root_position
        self._normalise_quaternions()

    def result(self):
        self.restore(self.best_state)
        self.positions[self.root] = self.root_position
        self._normalise_quaternions()
        return self.positions.copy(), self.orientations.copy()


def _part1_rotation_between(source, target, epsilon=1e-10):
    """Return the shortest rotation from source to target."""
    source_norm, target_norm = np.linalg.norm(source), np.linalg.norm(target)
    if source_norm < epsilon or target_norm < epsilon:
        return None
    source, target = source / source_norm, target / target_norm
    cross = np.cross(source, target)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cross_norm < epsilon:
        if dot > 0.0:
            return None
        basis = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, basis)
        return R.from_rotvec(axis / np.linalg.norm(axis) * np.pi)
    return R.from_rotvec(
        cross / cross_norm * np.arctan2(cross_norm, dot)
    )


def _part1_singularity_axis(context):
    """Choose a deterministic bend axis for a straight chain."""
    direction = context.target - context.positions[context.root]
    if np.linalg.norm(direction) < 1e-10:
        direction = context.positions[context.end] - context.positions[context.root]
    if np.linalg.norm(direction) < 1e-10:
        direction = context.positions[context.path[1]] - context.positions[context.root]
    direction /= max(np.linalg.norm(direction), 1e-10)
    basis = np.eye(3)[np.argmin(np.abs(direction))]
    axis = np.cross(direction, basis)
    return axis / max(np.linalg.norm(axis), 1e-10)


def _part1_break_singularity(context, angle=0.04):
    """Slightly rotate a reachable collinear chain out of singularity."""
    if len(context.path) < 3:
        return
    context.apply_world_rotation(
        context.path[0],
        R.from_rotvec(_part1_singularity_axis(context) * angle),
    )
    context.remember_best()


def _part1_solve_ccd(context, tolerance=0.005, max_iterations=80):
    """Align joints cyclically from the end effector back to the fixed root."""
    stagnant_rounds = recovery_count = 0
    for _ in range(max_iterations):
        if context.error() <= tolerance:
            break
        round_start = context.error()
        for joint in reversed(context.path[:-1]):
            pivot = context.positions[joint]
            delta = _part1_rotation_between(
                context.positions[context.end] - pivot,
                context.target - pivot,
            )
            if delta is not None:
                context.apply_world_rotation(joint, delta)
                context.remember_best()
            if context.error() <= tolerance:
                break

        improvement = round_start - context.error()
        stagnant_rounds = stagnant_rounds + 1 if improvement < 1e-8 else 0
        if stagnant_rounds >= 2:
            if context.reachable and recovery_count < 3:
                context.restore(context.best_state)
                sign = -1.0 if recovery_count % 2 else 1.0
                _part1_break_singularity(context, sign * 0.04)
                recovery_count += 1
                stagnant_rounds = 0
            else:
                break
    context.remember_best()
    return context.result()


def _part1_geometric_jacobian(context):
    """Build ball-joint columns with the lecture formula axis cross radius."""
    end_position = context.positions[context.end]
    columns = []
    for joint in context.path[:-1]:
        radius = end_position - context.positions[joint]
        columns.extend(np.cross(axis, radius) for axis in np.eye(3))
    return np.asarray(columns, dtype=np.float64).T


class _Part1GaussNewtonContext:
    """Gauss-Newton state using the original FK hierarchy directly."""

    def __init__(self, meta_data, joint_positions, joint_orientations, target_pose):
        self.meta_data = meta_data
        self.parents = list(meta_data.joint_parent)
        self.rest_positions = np.asarray(
            meta_data.joint_initial_position, dtype=np.float64
        )
        self.input_positions = np.asarray(joint_positions, dtype=np.float64).copy()
        self.input_orientations = np.asarray(
            joint_orientations, dtype=np.float64
        ).copy()
        self.requested_target = np.asarray(target_pose, dtype=np.float64).reshape(3)
        self.target = self.requested_target.copy()
        self.path = list(meta_data.get_path_from_root_to_end()[0])
        if len(self.path) < 2:
            raise ValueError("IK path must contain at least two joints.")
        if self.input_positions.shape != (len(self.parents), 3):
            raise ValueError("joint_positions has an invalid shape.")
        if self.input_orientations.shape != (len(self.parents), 4):
            raise ValueError("joint_orientations has an invalid shape.")
        if self.rest_positions.shape != self.input_positions.shape:
            raise ValueError("joint_initial_position has an invalid shape.")

        self.root = meta_data.joint_name.index(meta_data.root_joint)
        self.end = meta_data.joint_name.index(meta_data.end_joint)
        self.fixed_position = self.input_positions[self.root].copy()
        self.chain_length = self._path_chain_length()
        self.unreachable = self._preprocess_target()
        self.original_root = self._find_original_root()
        self.children = self._build_children()
        self.descendants = self._build_descendants()
        self.order = self._build_topological_order()
        self.variables = self._build_variable_joints()
        if not self.variables:
            raise ValueError("IK path must contain at least one movable edge.")

        self.local_matrices = self._global_quaternions_to_local_matrices()
        self.positions = np.zeros_like(self.input_positions)
        self.global_matrices = [np.eye(3) for _ in self.parents]
        self.forward_kinematics()
        self.best_error = self.error()
        self.best_state = self.snapshot()

    def _path_chain_length(self):
        return float(sum(
            np.linalg.norm(self.rest_positions[child] - self.rest_positions[parent])
            for parent, child in zip(self.path[:-1], self.path[1:])
        ))

    def _preprocess_target(self):
        direction = self.requested_target - self.fixed_position
        distance = np.linalg.norm(direction)
        if not np.isfinite(distance):
            self.target = self.fixed_position.copy()
            return True
        if distance <= self.chain_length or distance < 1e-12:
            self.target = self.requested_target.copy()
            return False

        margin = max(0.03, 0.05 * self.chain_length)
        stable_radius = max(self.chain_length - margin, 0.0)
        self.target = self.fixed_position + direction / distance * stable_radius
        return True

    def _find_original_root(self):
        for joint, parent in enumerate(self.parents):
            if parent == -1:
                return joint
        raise ValueError("The skeleton must contain one original root joint.")

    def _build_children(self):
        children = [[] for _ in self.parents]
        for child, parent in enumerate(self.parents):
            if parent != -1:
                children[parent].append(child)
        return children

    def _build_descendants(self):
        descendants = []
        for joint in range(len(self.parents)):
            seen, stack = set(), list(self.children[joint])
            while stack:
                child = stack.pop()
                seen.add(child)
                stack.extend(self.children[child])
            descendants.append(seen)
        return descendants

    def _build_topological_order(self):
        order, queue = [], [self.original_root]
        while queue:
            joint = queue.pop(0)
            order.append(joint)
            queue.extend(self.children[joint])
        if len(order) != len(self.parents):
            raise ValueError("The skeleton must be a connected tree.")
        return order

    def _build_variable_joints(self):
        variables = []
        for first, second in zip(self.path[:-1], self.path[1:]):
            if self.parents[first] == second:
                joint = second
            elif self.parents[second] == first:
                joint = first
            else:
                raise ValueError("IK path contains a non-adjacent edge.")
            if joint not in variables:
                variables.append(joint)
        return variables

    def _normalised_input_quaternions(self):
        quaternions = self.input_orientations.copy()
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        invalid = norms[:, 0] < 1e-12
        quaternions[~invalid] /= norms[~invalid]
        quaternions[invalid] = np.array([0.0, 0.0, 0.0, 1.0])
        return quaternions

    def _global_quaternions_to_local_matrices(self):
        quaternions = self._normalised_input_quaternions()
        global_matrices = [R.from_quat(quat).as_matrix() for quat in quaternions]
        local_matrices = [np.eye(3) for _ in self.parents]
        for joint in self.order:
            parent = self.parents[joint]
            if parent == -1:
                local_matrices[joint] = global_matrices[joint]
            else:
                local_matrices[joint] = global_matrices[parent].T @ global_matrices[joint]
        return local_matrices

    def snapshot(self):
        return [matrix.copy() for matrix in self.local_matrices]

    def restore(self, state):
        self.local_matrices = [matrix.copy() for matrix in state]
        self.forward_kinematics()

    def forward_kinematics(self):
        self.positions[...] = 0.0
        for joint in self.order:
            parent = self.parents[joint]
            if parent == -1:
                self.global_matrices[joint] = self.local_matrices[joint]
            else:
                self.global_matrices[joint] = (
                    self.global_matrices[parent] @ self.local_matrices[joint]
                )
            for child in self.children[joint]:
                offset = self.rest_positions[child] - self.rest_positions[joint]
                self.positions[child] = (
                    self.positions[joint] + self.global_matrices[joint] @ offset
                )

        # Keep the requested IK root fixed after solving in the original tree.
        self.positions += self.fixed_position - self.positions[self.root]

    def error(self):
        return float(np.linalg.norm(self.positions[self.end] - self.target))

    def remember_best(self):
        current_error = self.error()
        if current_error < self.best_error:
            self.best_error = current_error
            self.best_state = self.snapshot()

    def variable_scales(self):
        scales = []
        for joint in self.variables:
            name = self.meta_data.joint_name[joint]
            if name.endswith("_end") or "Toe" in name:
                scale = 0.04
            elif "Ankle" in name:
                scale = 0.08
            elif "Knee" in name:
                scale = 0.25
            elif "Hip" in name:
                scale = 0.45
            elif name in ("RootJoint", "pelvis_lowerback", "lowerback_torso"):
                scale = 0.8
            else:
                scale = 1.0
            scales.extend((scale, scale, scale))
        return np.asarray(scales, dtype=np.float64)

    def raw_point_derivative(self, joint, point, axis):
        if point not in self.descendants[joint]:
            return np.zeros(3, dtype=np.float64)
        return np.cross(axis, self.positions[point] - self.positions[joint])

    def jacobian(self):
        columns = []
        for joint in self.variables:
            for axis in np.eye(3):
                end_delta = self.raw_point_derivative(joint, self.end, axis)
                root_delta = self.raw_point_derivative(joint, self.root, axis)
                columns.append(end_delta - root_delta)
        return np.asarray(columns, dtype=np.float64).T

    def apply_parameter_step(self, step, scale, max_angle=0.2):
        parent_frames = {}
        for joint in self.variables:
            parent = self.parents[joint]
            parent_frames[joint] = (
                np.eye(3) if parent == -1 else self.global_matrices[parent].copy()
            )

        for joint, vector in zip(self.variables, step.reshape(-1, 3)):
            vector = vector * scale
            angle = np.linalg.norm(vector)
            if angle > max_angle:
                vector *= max_angle / angle
            if np.linalg.norm(vector) < 1e-12:
                continue

            # Convert a world-space rotation increment into the local frame.
            parent_global = parent_frames[joint]
            delta_world = R.from_rotvec(vector).as_matrix()
            delta_local = parent_global.T @ delta_world @ parent_global
            self.local_matrices[joint] = delta_local @ self.local_matrices[joint]
        self.forward_kinematics()

    def result(self):
        self.restore(self.best_state)
        quaternions = np.asarray([
            R.from_matrix(matrix).as_quat() for matrix in self.global_matrices
        ])
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        quaternions /= np.maximum(norms, 1e-12)
        self.positions[self.root] = self.fixed_position
        return self.positions.copy(), quaternions


def _part1_solve_gauss_newton(context, tolerance=0.005, max_iterations=100):
    """Damped Gauss-Newton solved directly in the original FK hierarchy."""
    if context.unreachable:
        tolerance = max(tolerance, 0.01)
        max_iterations = min(max_iterations, 20)

    damping, failed_steps, stagnant_steps = 1e-3, 0, 0
    last_error = context.error()
    column_scales = context.variable_scales()
    for _ in range(max_iterations):
        current_error = context.error()
        context.remember_best()
        if current_error <= tolerance:
            break
        if not np.isfinite(current_error):
            break

        jacobian = context.jacobian()
        residual = context.target - context.positions[context.end]
        if not (np.isfinite(jacobian).all() and np.isfinite(residual).all()):
            break

        weighted_jacobian = jacobian * column_scales[None, :]
        system = weighted_jacobian @ weighted_jacobian.T + damping * np.eye(3)
        try:
            z = weighted_jacobian.T @ np.linalg.solve(system, residual)
        except np.linalg.LinAlgError:
            z = weighted_jacobian.T @ np.linalg.pinv(system) @ residual
        step = column_scales * z
        step_norm = np.linalg.norm(step)
        if not np.isfinite(step_norm) or step_norm < 1e-10:
            break

        baseline, accepted = context.snapshot(), False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            context.restore(baseline)
            context.apply_parameter_step(step, scale)
            trial_error = context.error()
            if not np.isfinite(trial_error):
                continue
            if trial_error < current_error - 1e-10:
                damping = max(damping * 0.7, 1e-6)
                failed_steps, accepted = 0, True
                context.remember_best()
                break
        if accepted:
            improvement = last_error - context.error()
            stagnant_steps = stagnant_steps + 1 if improvement < 1e-6 else 0
            last_error = context.error()
            if stagnant_steps >= (3 if context.unreachable else 6):
                break
            continue

        context.restore(baseline)
        damping = min(damping * 10.0, 1e3)
        failed_steps += 1
        if failed_steps >= (3 if context.unreachable else 6):
            break

    context.remember_best()
    return context.result()


def _part1_torch_rotation_matrices(torch, rotation_vectors):
    """Differentiable Rodrigues formula using sinc for stability near zero."""
    count = rotation_vectors.shape[0]
    x, y, z = rotation_vectors.unbind(dim=1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=1
    ).reshape(count, 3, 3)
    angles = torch.linalg.norm(rotation_vectors, dim=1)
    first = torch.sinc(angles / np.pi)
    second = 0.5 * torch.sinc(angles / (2.0 * np.pi)) ** 2
    identity = torch.eye(
        3, dtype=rotation_vectors.dtype, device=rotation_vectors.device
    ).expand(count, 3, 3)
    return identity + first[:, None, None] * skew + second[:, None, None] * (skew @ skew)


def _part1_torch_end_position(torch, root, offsets, parameters):
    """Run differentiable forward kinematics on the rerooted chain."""
    matrices = _part1_torch_rotation_matrices(torch, parameters)
    position = root
    accumulated = torch.eye(3, dtype=parameters.dtype, device=parameters.device)
    for index in range(offsets.shape[0]):
        accumulated = accumulated @ matrices[index]
        position = position + accumulated @ offsets[index]
    return position


def _part1_apply_local_parameters(context, parameters):
    """Convert optimized local rotations to world rotations and write pose."""
    accumulated = np.eye(3)
    for joint, vector in zip(context.path[:-1], parameters):
        local = R.from_rotvec(vector).as_matrix()
        world = accumulated @ local @ accumulated.T
        context.apply_world_rotation(joint, R.from_matrix(world))
        accumulated = accumulated @ local






# Refine autograd initialization: preserve the unmodified pose as the baseline
# and add a bend only when the geometric gradient is singular.
def _part1_solve_autograd(context, tolerance=0.005, max_iterations=400):
    """PyTorch autograd with Adam and a safe zero-rotation baseline."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch autograd IK requires PyTorch. Install torch or switch "
            "part1_inverse_kinematics to CCD/Gauss-Newton."
        ) from error

    dtype = torch.float64
    root = torch.tensor(context.positions[context.root], dtype=dtype)
    target = torch.tensor(context.target, dtype=dtype)
    offsets = torch.tensor(
        np.asarray([
            context.positions[child] - context.positions[parent]
            for parent, child in zip(context.path[:-1], context.path[1:])
        ]),
        dtype=dtype,
    )
    zero_parameters = np.zeros((len(context.path) - 1, 3), dtype=np.float64)
    initial = zero_parameters.copy()
    residual = context.positions[context.end] - context.target
    gradient = _part1_geometric_jacobian(context).T @ residual
    if (
        context.reachable
        and context.error() > tolerance
        and np.linalg.norm(gradient) < 1e-10
        and len(context.path) >= 3
    ):
        index = min(len(initial) - 1, len(initial) // 2)
        initial[index] = _part1_singularity_axis(context) * 0.04

    parameters = torch.tensor(initial, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=0.06)
    best_parameters = torch.tensor(zero_parameters, dtype=dtype)
    best_error = context.error()
    for _ in range(max_iterations):
        optimizer.zero_grad()
        end_position = _part1_torch_end_position(torch, root, offsets, parameters)
        position_error = end_position - target
        current_error = float(torch.linalg.norm(position_error).detach())
        if current_error < best_error:
            best_error = current_error
            best_parameters = parameters.detach().clone()
        if current_error <= tolerance:
            break
        loss = 0.5 * torch.dot(position_error, position_error)
        loss = loss + 1e-7 * torch.sum(parameters * parameters)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_end = _part1_torch_end_position(torch, root, offsets, parameters)
        if float(torch.linalg.norm(final_end - target)) < best_error:
            best_parameters = parameters.detach().clone()
    _part1_apply_local_parameters(context, best_parameters.cpu().numpy())
    context.remember_best()
    return context.result()

def solve_ccd(meta_data, joint_positions, joint_orientations, target_pose):
    """Solve Part1 IK with cyclic coordinate descent."""
    context = _Part1IKContext(
        meta_data, joint_positions, joint_orientations, target_pose
    )
    return _part1_solve_ccd(context)


def solve_autograd(meta_data, joint_positions, joint_orientations, target_pose):
    """Solve Part1 IK with PyTorch automatic differentiation."""
    context = _Part1IKContext(
        meta_data, joint_positions, joint_orientations, target_pose
    )
    return _part1_solve_autograd(context)


def solve_gauss_newton(
    meta_data, joint_positions, joint_orientations, target_pose
):
    """Solve Part1 IK with the damped Gauss-Newton method."""
    context = _Part1GaussNewtonContext(
        meta_data, joint_positions, joint_orientations, target_pose
    )
    return _part1_solve_gauss_newton(context)
