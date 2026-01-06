import torch

class BatchQuatOperations:
    def __init__(self, device='cuda'):
        self.device = device

    def quat_mult(self, a, b):
        aw, ax, ay, az = torch.unbind(a, -1)
        bw, bx, by, bz = torch.unbind(b, -1)
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw

        return torch.stack((ow, ox, oy, oz), -1)

    def quat_conjugate(self, q):
        return q * torch.tensor([1, -1, -1, -1], device=self.device)

    def compute_relative_quat(self, q1, q2):
        q2_inv = self.quat_conjugate(q2)
        q_relative = self.quat_mult(q1, q2_inv)
        norm = torch.norm(q_relative, dim=-1, keepdim=True)
        return q_relative / (norm + 1e-8)

    def batch_angular_error(self, pred, target):
        q_rel = self.compute_relative_quat(pred, target)
        real_part = q_rel[..., 0].clamp(-1 + 1e-6, 1 - 1e-6)
        return 2 * torch.acos(torch.abs(real_part))