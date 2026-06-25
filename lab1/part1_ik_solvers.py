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


def _part1_apply_parameter_step(context, step, scale, max_angle=0.25):
    """Apply one world-space rotation-vector step to all chain joints."""
    for joint, vector in reversed(list(zip(context.path[:-1], step.reshape(-1, 3)))):
        vector = vector * scale
        angle = np.linalg.norm(vector)
        if angle > max_angle:
            vector *= max_angle / angle
        if np.linalg.norm(vector) > 1e-12:
            context.apply_world_rotation(joint, R.from_rotvec(vector))


def _part1_solve_gauss_newton(context, tolerance=0.005, max_iterations=80):
    """Damped Gauss-Newton using a stable task-space normal equation."""
    damping, failed_steps = 1e-3, 0
    for _ in range(max_iterations):
        current_error = context.error()
        if current_error <= tolerance:
            break
        jacobian = _part1_geometric_jacobian(context)
        residual = context.target - context.positions[context.end]
        system = jacobian @ jacobian.T + damping * np.eye(3)
        try:
            step = jacobian.T @ np.linalg.solve(system, residual)
        except np.linalg.LinAlgError:
            step = jacobian.T @ np.linalg.pinv(system) @ residual

        baseline, accepted = context.snapshot(), False
        for scale in (1.0, 0.5, 0.25, 0.125):
            context.restore(baseline)
            _part1_apply_parameter_step(context, step, scale)
            if context.error() < current_error - 1e-10:
                context.remember_best()
                damping = max(damping * 0.7, 1e-6)
                failed_steps, accepted = 0, True
                break
        if accepted:
            continue

        context.restore(baseline)
        damping = min(damping * 10.0, 1e3)
        failed_steps += 1
        if context.reachable and failed_steps <= 3:
            sign = -1.0 if failed_steps % 2 == 0 else 1.0
            _part1_break_singularity(context, sign * 0.04)
        else:
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
    context = _Part1IKContext(
        meta_data, joint_positions, joint_orientations, target_pose
    )
    return _part1_solve_gauss_newton(context)
