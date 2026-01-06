import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import random
from typing import List, Optional


class Sim_Env(gym.Env):
    def __init__(self):
        super(Sim_Env, self).__init__()
        self.state_dim = 13
        self.action_dim = 6
        self.dt = 0.1
        high_obs = np.array([130.0, 130.0, 130.0, 7.0, 7.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5], dtype=np.float32)
        low_obs = np.array([-130.0, -130.0, -130.0, -7.0, -7.0, -7.0, -1.0, -1.0, -1.0, -1.0, -1.5, -1.5, -1.5],
                           dtype=np.float32)
        self.observation_space = spaces.Box(high=high_obs, low=low_obs)
        high_action = np.array([20.0, 20.0, 20.0, 10.0, 10.0, 10.0], dtype=np.float32)
        low_action = np.array([-20.0, -20.0, -20.0, -10.0, -10.0, -10.0], dtype=np.float32)
        self.action_space = spaces.Box(high=high_action, low=low_action)
        self.GM = 39860047 * 10 ** 7
        self.orbit_data_chief = np.array([7500000.0, 0.0, 45.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.radius = 7500000.0
        mu = 39860047 * 10 ** 7
        self.n = np.sqrt(mu / (self.radius ** 3))
        self.deputy_mass = 100.0
        self.chief_mass = 100.0
        self.deputy_len = 2.52
        self.deputy_wid = self.deputy_h = 1.26
        self.deputy_I = self.inertia_matrix(self.deputy_len, self.deputy_wid, self.deputy_h)
        self.deputy_I_inverse = np.linalg.inv(self.deputy_I)  # NumPy 矩阵求逆替换 torch.inverse
        self.chief_docking_port_pos = np.array([0, 0.63, 0], dtype=np.float32)
        self.deputy_docking_port_pos = np.array([0, -0.63, 0], dtype=np.float32)
        self.target_state = np.array([0.5, 1.26, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.pre_reward = np.zeros(13, dtype=np.float32)
        self.cur_reward = np.zeros(13, dtype=np.float32)
        self.pos_threshold = 0.05
        self.vel_threshold = 0.01
        self.att_threshold = 0.01
        self.omega_threshold = 0.01

        self.pos_error_max = 50.0
        self.vel_error_max = 3.0
        self.att_error_max = np.pi / 2
        self.omega_error_max = 1.0
        self.pos_weight = 0.8
        self.vel_weight = 0.4
        self.att_weight = 4.0
        self.omega_weight = 1.8

        self.action_penalty_coeff = 0.0008

        self.convergence_rewards = [1.0, 1.0, 4.0, 2.0]
        self.done_reward = 80.0
        self.diverge_penalty = -80.0
        self.timeout_penalty = -30.0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.done = False
        self.chief_pos, self.chief_vel = self.classic_coord(self.orbit_data_chief, self.GM)
        self.chief_info = np.concatenate((self.chief_pos, self.chief_vel), axis=0)
        self.orbit_data_deputy, distance = self.adjust_orbit_elements(self.orbit_data_chief, self.GM)
        while distance < 75 or distance > 125:
            self.orbit_data_deputy, distance = self.adjust_orbit_elements(self.orbit_data_chief, self.GM)
        self.deputy_pos, self.deputy_vel = self.classic_coord(self.orbit_data_deputy, self.GM)

        self.q_chief = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.q_deputy = self.random_quaternion(1).squeeze(0).astype(np.float32)
        self.deputy_angv = np.random.rand(3).astype(np.float32)
        self.chief_angv = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.q_deputy = self._normalize_quaternion(self.q_deputy)
        self.q_chief = self._normalize_quaternion(self.q_chief)
        self.deputy_attitude = np.concatenate((self.q_deputy, self.deputy_angv), axis=0)
        self.chief_attitude = np.concatenate((self.q_chief, self.chief_angv), axis=0)
        self.deputy_rmatrix = self.quaternion_to_matrix(self.q_deputy)
        self.chief_rmatrix = self.quaternion_to_matrix(self.q_chief)

        self.q_rel = self.compute_relative_quat(self.q_chief, self.q_deputy)
        self.angv_rel = (self.deputy_angv - self.chief_angv).astype(np.float32)
        self.relative_pos, self.relative_vel = self.eci_to_lvlh(self.chief_pos, self.chief_vel, self.deputy_pos,
                                                                self.deputy_vel)

        self.relative_info = np.concatenate((self.relative_pos, self.relative_vel), axis=0)
        self.obs = np.concatenate((self.relative_info, self.q_rel, self.angv_rel), axis=0).astype(
            np.float32)
        self.input = np.concatenate((self.relative_info, self.deputy_attitude))
        self.step_count = 0
        return self.obs, {}

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = action
        else:
            action = np.array(action, dtype=np.float32)

        self.pos_f = action[:3]
        self.ang_t = action[3:]
        min_dt = np.array(1e-6, dtype=np.float32)
        pre_state = self.input
        self.input = self.rk4_step_cw(self.input, self.dt, action)
        self.relative_info = self.input[:6]
        self.q_deputy = self.input[6:10]
        self.deputy_angv = self.input[10:]
        self.q_deputy = self._normalize_quaternion(self.q_deputy)
        self.input[6:10] = self.q_deputy
        self.deputy_attitude = np.concatenate((self.q_deputy, self.deputy_angv))
        self.deputy_rmatrix = self.quaternion_to_matrix(self.q_deputy)

        self.q_rel = self.compute_relative_quat(self.q_chief, self.q_deputy)
        self.angv_rel = (self.deputy_angv - self.chief_angv).astype(np.float32)
        self.obs = np.concatenate((self.relative_info, self.q_rel, self.angv_rel), axis=0).astype(np.float32)

        current_pos = self.input[:3]
        target_pos = self.target_state[:3]
        current_vel = self.input[3:6]
        target_vel = self.target_state[3:6]
        current_quat = self.input[6:10]
        target_quat = self.target_state[6:10]
        current_omega = self.input[10:]
        target_omega = self.target_state[10:]

        pos_error = np.linalg.norm(current_pos - target_pos)
        pos_error = np.clip(pos_error, 0, self.pos_error_max)

        vel_error = np.linalg.norm(current_vel - target_vel)
        vel_error = np.clip(vel_error, 0, self.vel_error_max)

        q_rel = self.compute_relative_quat(target_quat, current_quat)
        att_error = 2 * np.arccos(np.clip(np.abs(q_rel[0]), 0, 1))
        att_error = np.clip(att_error, 0, self.att_error_max)

        omega_error = np.linalg.norm(current_omega - target_omega)
        omega_error = np.clip(omega_error, 0, self.omega_error_max)
        base_penalty = (self.pos_weight * pos_error ** 2 +
                        self.vel_weight * vel_error ** 2 +
                        self.att_weight * att_error ** 2 +
                        self.omega_weight * omega_error ** 2)

        action_norm = np.linalg.norm(action)
        action_penalty = self.action_penalty_coeff * action_norm ** 2

        convergence_reward = 0.0
        if pos_error < self.pos_threshold:
            convergence_reward += self.convergence_rewards[0]
        if vel_error < self.vel_threshold:
            convergence_reward += self.convergence_rewards[1]
        if att_error < self.att_threshold:
            convergence_reward += self.convergence_rewards[2]
        if omega_error < self.omega_threshold:
            convergence_reward += self.convergence_rewards[3]

        done_reward = 0.0
        if (pos_error < self.pos_threshold and
                vel_error < self.vel_threshold and
                att_error < self.att_threshold and
                omega_error < self.omega_threshold):
            done_reward = self.done_reward
            self.done = True

        reward = -base_penalty - action_penalty + convergence_reward + done_reward

        self.step_count += 1
        if (np.linalg.norm(current_pos) > 200 or
                np.linalg.norm(current_vel) > 9 or
                np.linalg.norm(current_omega) > 20):
            self.done = True
            reward += self.diverge_penalty

        return self.obs, reward, self.done, False, {}

    def inertia_matrix(self, l, w, h):
        ix = self.deputy_mass * (w ** 2 + h ** 2) / 12
        iy = self.deputy_mass * (l ** 2 + h ** 2) / 12
        iz = self.deputy_mass * (l ** 2 + w ** 2) / 12
        deputy_I = np.diag(np.array([ix, iy, iz], dtype=np.float32))
        return deputy_I

    def classic_coord(self, data, GM):
        a = data[0]
        e = 0
        i = np.deg2rad(data[2])
        w = 0
        W = np.deg2rad(data[4])
        fai = np.deg2rad(data[5])
        u = fai + w

        Coordinate = (a / (1 + e * np.cos(fai))) * np.array([
            np.cos(W) * np.cos(u) - np.sin(W) * np.sin(u) * np.cos(i),
            np.sin(W) * np.cos(u) + np.cos(W) * np.sin(u) * np.cos(i),
            np.sin(i) * np.sin(u)
        ], dtype=np.float32)

        V = np.sqrt(GM / a) * np.array([
            -np.cos(W) * (np.sin(u) + e * np.sin(w)) - np.sin(W) * (np.cos(u) + e * np.cos(w)) * np.cos(i),
            -np.sin(W) * (np.sin(u) + e * np.sin(w)) + np.cos(W) * (np.cos(u) + e * np.cos(w)) * np.cos(i),
            np.sin(i) * (np.cos(u) + e * np.cos(w))
        ], dtype=np.float32)

        return Coordinate, V

    def random_quaternion(self, n: int):
        o = np.random.randn(n, 4).astype(np.float32)
        s = np.sum(o * o, axis=1)
        o = o / np.copysign(np.sqrt(s), o[:, 0])[:, None]
        return o

    def _normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        return np.where(norm < 1e-12, np.array([1.0, 0, 0, 0], dtype=np.float32), q / norm)

    def quaternion_to_matrix(self, quaternions):
        r, i, j, k = quaternions[0], quaternions[1], quaternions[2], quaternions[3]
        two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
        o = np.stack([
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j)
        ], axis=-1)

        return o.reshape(quaternions.shape[:-1] + (3, 3)).astype(np.float32)

    def quat_mult(self, a, b):
        aw, ax, ay, az = a[0], a[1], a[2], a[3]
        bw, bx, by, bz = b[0], b[1], b[2], b[3]

        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw

        return np.stack((ow, ox, oy, oz), axis=-1).astype(np.float32)

    def quat_conjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    def compute_relative_quat(self, q1, q2):
        q1_inv = self.quat_conjugate(q1)
        q_relative = self.quat_mult(q2, q1_inv)
        q_relative = q_relative / np.linalg.norm(q_relative)
        return q_relative

    def eci_to_lvlh(self, r_chief_eci, v_chief_eci, r_deputy_eci, v_deputy_eci):
        r_norm = np.linalg.norm(r_chief_eci)
        o_r = r_chief_eci / r_norm
        h_vec = np.cross(r_chief_eci, v_chief_eci, axis=0)
        h_norm = np.linalg.norm(h_vec)
        o_h = h_vec / h_norm

        o_t = np.cross(o_h, o_r, axis=0)
        R_eci_to_lvlh = np.stack([o_r, o_t, o_h], axis=0)
        dr_eci = r_deputy_eci - r_chief_eci
        r_lvlh = np.matmul(R_eci_to_lvlh, dr_eci)

        omega = h_norm / (r_norm ** 2)
        omega_lvlh = np.array([0, 0, omega], dtype=np.float32)

        dv_eci = v_deputy_eci - v_chief_eci
        v_temp = np.matmul(R_eci_to_lvlh, dv_eci)
        v_lvlh = v_temp - np.cross(omega_lvlh, r_lvlh, axis=0)

        return r_lvlh, v_lvlh

    def adjust_orbit_elements(self, target_elements, GM):
        a0, e0, i0, omega0, Omega0, nu0 = target_elements

        a_min = 0.0167
        a_max = 0.0500
        angle_min = 0.0003438
        angle_max = 0.000400

        a_adjust = (np.random.rand() > 0.5) * (a_min + (a_max - a_min) * np.random.rand()) - \
                   (np.random.rand() <= 0.5) * (a_min + (a_max - a_min) * np.random.rand())
        angle_adjust = angle_min + (angle_max - angle_min) * np.random.rand()
        adjusted_elements = target_elements.copy()
        adjusted_elements[0] = a0 + a_adjust

        if np.random.rand() > 0.5:
            sign = 1
        else:
            sign = -1
        adjusted_elements[2] = i0 + sign * angle_adjust

        if np.random.rand() > 0.5:
            sign = 1
        else:
            sign = -1
        adjusted_elements[4] = Omega0 + sign * angle_adjust * 1.1

        if np.random.rand() > 0.5:
            sign = 1
        else:
            sign = -1
        adjusted_elements[5] = nu0 + sign * angle_adjust * 0.9
        Coord1, _ = self.classic_coord(target_elements, GM)
        Coord2, _ = self.classic_coord(adjusted_elements, GM)
        distance = np.linalg.norm(Coord1 - Coord2)

        return adjusted_elements, distance

    def spacecraft_dynamic(self, s, u):
        x, y, z, vx, vy, vz = s[:6]
        fx = u[0]
        fy = u[1]
        fz = u[2]

        Fx = fx / self.deputy_mass
        Fy = fy / self.deputy_mass
        Fz = fz / self.deputy_mass
        ax = Fx - 2 * self.n * vy
        ay = Fy + 2 * self.n * vx + 3 * y * self.n ** 2
        az = Fz - z * self.n ** 2
        dpdt = np.array([vx, vy, vz, ax, ay, az], dtype=np.float32)
        q = s[6:10].astype(np.float32)
        norm_q = np.linalg.norm(q)
        if norm_q < 1e-12:
            q = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            q = q / norm_q

        omega = s[10:].reshape(-1, 1).astype(np.float32)
        torque = u[3:].reshape(-1, 1).astype(np.float32)

        torque = np.matmul(self.quaternion_to_matrix(q), torque)
        omega_quat = np.array([0, omega[0, 0], omega[1, 0], omega[2, 0]], dtype=np.float32)
        dqdt = 0.5 * self.quat_mult(q, omega_quat)
        skew_omega_b = self.skew_symmetric_matrix(omega)
        domega = np.matmul(
            self.deputy_I_inverse,
            (torque - np.matmul(np.matmul(skew_omega_b, self.deputy_I), omega))
        ).squeeze()
        dydt = np.concatenate((dpdt, dqdt, domega), axis=0)
        return dydt

    def skew_symmetric_matrix(self, x):
        x = x.reshape(3)
        x1, x2, x3 = x[0], x[1], x[2]
        return np.array([
            [0, -x3, x2],
            [x3, 0, -x1],
            [-x2, x1, 0]
        ], dtype=np.float32)

    def rk4_step_cw(self, s, dt, u):
        k1 = self.spacecraft_dynamic(s, u)
        k2 = self.spacecraft_dynamic(s + dt * k1 / 2, u)
        k3 = self.spacecraft_dynamic(s + dt * k2 / 2, u)
        k4 = self.spacecraft_dynamic(s + dt * k3, u)
        y_next = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next.astype(np.float32)