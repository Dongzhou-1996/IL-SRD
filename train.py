import torch
import numpy as np
import os
from ACT import ACT
from dataloader import Dataloader, normalization, denormalization
import csv
from sim_env import Sim_Env
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from batchquaterror import BatchQuatOperations
import time

class train_IL_SRD(object):
    def __init__(self, hidden_dim, seq_len, input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, activation, normalize_before, return_intermediate_dec, ep_num, seed, index,
                 batch_size, lr, weight_decay):
        super().__init__()
        self.action_std1 = None
        self.action_std2 = None
        self.action_mean1 = None
        self.action_mean2 = None
        self.state_std1 = None
        self.state_std2 = None
        self.state_std3 = None
        self.state_mean1 = None
        self.state_mean2 = None
        self.state_mean3 = None
        self.k = 1 / seq_len
        self.device = torch.device('cuda')
        self.policy = ACT(hidden_dim, seq_len, input_dim, output_dim, nhead, num_encoder_layers,
                          num_decoder_layers, dim_feedforward, dropout, activation,
                          normalize_before, return_intermediate_dec).to(self.device)

        self.dataloader = Dataloader(seq_len)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.normalize_before = normalize_before
        self.return_intermediate_dec = return_intermediate_dec

        self.ep_num = ep_num
        self.seed = seed
        self.index = index
        self.train_batch_size = batch_size
        self.val_batch_size = 128
        self.lr = lr
        self.weight_decay = weight_decay
        self.norm_method = 'mu'
        self.suffix = None
        self.seq_len = seq_len
        self.ckpt_dir = None
        self.eval_env = None
        self.model_dir = '.\\model_dir\\'
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), self.lr, weight_decay=self.weight_decay)
        self.batch_quat = BatchQuatOperations()
        # seed = np.random.randint(1, 10000)
        self.env = Sim_Env()
        self.norm_dict = {}
        self.max_incre = None
        self.min_incre = None
        # self.loss_fn = HausdorffDistanceLoss()

    def train(self, loss_fn):
        self.optimizer.zero_grad()
        self.loss_list = []
        previous_epoch = None
        self.suffix = f'{self.hidden_dim}_{self.dim_feedforward}_{self.ep_num}_{self.nhead}_{self.num_encoder_layers}' \
                      f'_{self.num_decoder_layers}_{self.dropout}_{self.seed}_{self.lr}_{self.weight_decay}' \
                      f'_{self.train_batch_size}_{self.seq_len}_{self.k}_{loss_fn}_{self.norm_method}_less_noise'

        os.makedirs(f'.\\model_set\\{self.suffix}', exist_ok=True)
        self.loss_pos_list = []
        self.loss_vel_list = []
        self.loss_attitude_list = []
        self.loss_omega_list = []
        self.loss_kl_list = []

        for epoch in range(self.ep_num):
            train_dict, val_dict, self.norm_dict = self.dataloader.prepare_data(
                50, 0.8, self.train_batch_size, self.val_batch_size, self.seq_len, self.norm_method)
            states = train_dict['state']
            actions = train_dict['action']
            is_pad = train_dict['is_pad']
            self.max_incre = self.norm_dict['incre']['max']
            self.min_incre = self.norm_dict['incre']['min']
            action_max = self.norm_dict['local_action_max']
            action_min = self.norm_dict['local_action_min']
            self.incre_limit_eval = torch.cat(
                (self.max_incre.unsqueeze(0).unsqueeze(1), self.min_incre.unsqueeze(0).unsqueeze(1)), dim=0)
            pred_actions, mu, sigma = self.policy.forward(states, actions, is_pad)

            overall_pred_actions = torch.cat((states.unsqueeze(1), pred_actions), dim=1)
            pred_actions = torch.clamp(pred_actions, action_min.unsqueeze(1), action_max.unsqueeze(1))
            increment = overall_pred_actions[:, 1:, :] - overall_pred_actions[:, :-1, :]
            origin_pred_actions = denormalization(self.norm_method, pred_actions, self.norm_dict['action'])
            origin_actions = denormalization(self.norm_method, actions, self.norm_dict['action'])

            loss_pos1 = F.mse_loss(pred_actions[:, :, :3] * ~is_pad.unsqueeze(-1),
                                   actions[:, :, :3].clone() * ~is_pad.unsqueeze(-1), reduction='sum') / (
                            ~is_pad).sum()

            loss_pos2 = F.mse_loss(origin_pred_actions[:, :, :3] * ~is_pad.unsqueeze(-1),
                                   origin_actions[:, :, :3].clone() * ~is_pad.unsqueeze(-1), reduction='sum') / (
                            ~is_pad).sum()

            loss_pos = (0.2 * loss_pos1 + 0.8 * loss_pos2)

            loss_vel1 = F.mse_loss(pred_actions[:, :, 3:6] * ~is_pad.unsqueeze(-1),
                                   actions[:, :, 3:6].clone() * ~is_pad.unsqueeze(-1), reduction='sum') / (
                            ~is_pad).sum() * 8

            loss_vel2 = F.mse_loss(origin_pred_actions[:, :, 3:6] * ~is_pad.unsqueeze(-1),
                                   origin_actions[:, :, 3:6].clone() * ~is_pad.unsqueeze(-1), reduction='sum') / (
                            ~is_pad).sum() * 8

            loss_vel = (0.5 * loss_vel1 + 0.5 * loss_vel2) * 8

            loss_omega1 = F.mse_loss(pred_actions[:, :, 10:] * ~is_pad.unsqueeze(-1),
                                     actions[:, :, 10:].clone() * ~is_pad.unsqueeze(-1), reduction='sum') / (
                              ~is_pad).sum() * 10

            loss_omega2 = F.mse_loss(origin_pred_actions[:, :, 10:] * ~is_pad.unsqueeze(-1),
                                     origin_actions[:, :, 10:].clone() * ~is_pad.unsqueeze(-1),
                                     reduction='sum') / (
                              ~is_pad).sum() * 10

            loss_omega = (loss_omega1 * 0.8 + 0.2 * loss_omega2) / 3

            theta = self.batch_quat.batch_angular_error(pred_actions[:, :, 6:10] * ~is_pad.unsqueeze(-1),
                                                        actions[:, :, 6:10] * ~is_pad.unsqueeze(
                                                            -1).clone()).sum() / (~is_pad).sum()

            # torch.autograd.set_detect_anomaly(True)
            loss_attitude = theta ** 2 * 6 * 20
            loss = loss_pos + loss_attitude + loss_omega + loss_vel
            total_kld = kl_divergence(mu, sigma)
            loss_kl = 4 * total_kld[0]
            loss = loss + loss_kl
            self.loss_kl_list.append(loss_kl.detach().clone().cpu())

            self.loss_pos_list.append(loss_pos.detach().clone().cpu())
            self.loss_vel_list.append(loss_vel.detach().clone().cpu())
            self.loss_omega_list.append(loss_omega.detach().clone().cpu())
            self.loss_attitude_list.append(loss_attitude.detach().clone().cpu())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_list.append(loss)
            print(f'\rThe loss is: {loss}, current epoch is: {epoch}')
        t = np.linspace(1, self.ep_num, self.ep_num)
        self.loss_pos_list = np.array(self.loss_pos_list)
        self.loss_vel_list = np.array(self.loss_vel_list)
        self.loss_omega_list = np.array(self.loss_omega_list)
        self.loss_attitude_list = np.array(self.loss_attitude_list)
        plt.figure(figsize=(14, 10))
        plt.plot(t, self.loss_pos_list, label='loss_pos', color='red')
        plt.plot(t, self.loss_vel_list, label='loss_vel', color='blue')
        plt.plot(t, self.loss_omega_list, label='loss_omega', color='green')
        plt.plot(t, self.loss_attitude_list, label='loss_attitude', color='yellow')
        self.loss_kl_list = np.array(self.loss_kl_list)
        plt.plot(t, self.loss_kl_list, label='loss_kl', color='cyan')
        plt.legend()
        plt.show()

        self.ckpt_dir = os.path.join(f'.\\model_set\\{self.suffix}', 'model.pth')
        if os.path.exists(self.ckpt_dir):
            os.remove(self.ckpt_dir)
        torch.save(self.policy.state_dict(), self.ckpt_dir)

    def eval_new(self, max_timesteps, seed):
        self.policy = ACT(self.hidden_dim, self.seq_len, self.input_dim, self.output_dim, self.nhead,
                          self.num_encoder_layers, self.num_decoder_layers, self.dim_feedforward, self.dropout,
                          self.activation, self.normalize_before, self.return_intermediate_dec).to(self.device)
        self.policy.eval()
        self.policy.load_state_dict(
            torch.load(
                f'.\\model_set\\{self.suffix}\\model.pth'))
        all_time_actions = torch.zeros([max_timesteps, max_timesteps + self.seq_len, self.output_dim],
                                       device=torch.device('cuda'))
        obs, _ = self.env.reset(seed)
        time_list = []
        start_time = time.time()
        obs = torch.tensor(obs).to(torch.float32).to(self.device).unsqueeze(0)
        act_input, _ = normalization(self.norm_method, obs.detach().clone(), self.norm_dict['state'])
        act_input = torch.tensor(act_input.detach().clone().requires_grad_(True), device=self.device).to(
            torch.float32)
        actions = np.zeros((max_timesteps + 1, 13))
        actions[0, :] = obs.detach().cpu().numpy()
        relative_x = [obs[:, 0].detach().cpu().numpy()]
        relative_y = [obs[:, 1].detach().cpu().numpy()]
        relative_z = [obs[:, 2].detach().cpu().numpy()]
        relative_vx = [obs[:, 3].detach().cpu().numpy()]
        relative_vy = [obs[:, 4].detach().cpu().numpy()]
        relative_vz = [obs[:, 5].detach().cpu().numpy()]
        relative_qw = [obs[:, 6].detach().cpu().numpy()]
        relative_qx = [obs[:, 7].detach().cpu().numpy()]
        relative_qy = [obs[:, 8].detach().cpu().numpy()]
        relative_qz = [obs[:, 9].detach().cpu().numpy()]
        relative_omegax = [obs[:, 10].detach().cpu().numpy()]
        relative_omegay = [obs[:, 11].detach().cpu().numpy()]
        relative_omegaz = [obs[:, 12].detach().cpu().numpy()]
        with torch.inference_mode():
            for t in range(max_timesteps):
                all_actions, _, _ = self.policy(act_input, actions=None)
                all_time_actions[[t], t:t + self.seq_len] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                weights = np.exp(-self.k * np.arange(len(actions_for_curr_step)))
                weights = weights / weights.sum()
                weights = torch.from_numpy(weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * weights).sum(dim=0, keepdim=True)
                current_state = denormalization(self.norm_method, raw_action, self.norm_dict['action'])
                current_incre = torch.clamp((current_state - obs), self.min_incre, self.max_incre)
                obs = obs + current_incre
                obs[:, 6:10] = obs[:, 6:10] / torch.norm(obs[:, 6:10], dim=1)

                actions[t + 1, :] = obs.cpu().numpy()
                relative_x.append(obs[:, 0].detach().cpu().numpy())
                relative_y.append(obs[:, 1].detach().cpu().numpy())
                relative_z.append(obs[:, 2].detach().cpu().numpy())
                relative_vx.append(obs[:, 3].detach().cpu().numpy())
                relative_vy.append(obs[:, 4].detach().cpu().numpy())
                relative_vz.append(obs[:, 5].detach().cpu().numpy())
                relative_qw.append(obs[:, 6].detach().cpu().numpy())
                relative_qx.append(obs[:, 7].detach().cpu().numpy())
                relative_qy.append(obs[:, 8].detach().cpu().numpy())
                relative_qz.append(obs[:, 9].detach().cpu().numpy())
                relative_omegax.append(obs[:, 10].detach().cpu().numpy())
                relative_omegay.append(obs[:, 11].detach().cpu().numpy())
                relative_omegaz.append(obs[:, 12].detach().cpu().numpy())
                act_input, _ = normalization(self.norm_method, obs.detach().clone(), self.norm_dict['state'])
                act_input = torch.tensor(act_input.detach().clone().requires_grad_(True), device=self.device).to(
                    torch.float32)

        end_time = time.time()
        t = np.linspace(1, len(relative_x), len(relative_x))
        data = pd.DataFrame({
            'step': t,
            'x': relative_x,
            'y': relative_y,
            'z': relative_z,
            'vx': relative_vx,
            'vy': relative_vy,
            'vz': relative_vz,
            'qw': relative_qw,
            'qx': relative_qx,
            'qy': relative_qy,
            'qz': relative_qz,
            'omegax': relative_omegax,
            'omegay': relative_omegay,
            'omegaz': relative_omegaz,
        })

        fig = plt.figure(figsize=(14, 10))

        gs = GridSpec(2, 2, height_ratios=[1, 1])
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])

        ax1.plot(t, data['x'], label='X', linewidth=1.5, color='#E24A33')  # 红色调
        ax1.plot(t, data['y'], label='Y', linewidth=1.5, color='#348ABD')  # 蓝色调
        ax1.plot(t, data['z'], label='Z', linewidth=1.5, color='#988ED5')  # 紫色调

        ax2.plot(t, data['vx'], label='X', linewidth=1.5, color='#E24A33')  # 红色调
        ax2.plot(t, data['vy'], label='Y', linewidth=1.5, color='#348ABD')  # 蓝色调
        ax2.plot(t, data['vz'], label='Z', linewidth=1.5, color='#988ED5')  # 紫色调

        ax3.plot(t, data['qw'], label='w', linewidth=1.5, color='green')  # 红色调
        ax3.plot(t, data['qx'], label='X', linewidth=1.5, color='#E24A33')  # 红色调
        ax3.plot(t, data['qy'], label='Y', linewidth=1.5, color='#348ABD')  # 蓝色调
        ax3.plot(t, data['qz'], label='Z', linewidth=1.5, color='#988ED5')  # 紫色调

        ax4.plot(t, data['omegax'], label='X', linewidth=1.5, color='#E24A33')  # 红色调
        ax4.plot(t, data['omegay'], label='Y', linewidth=1.5, color='#348ABD')  # 蓝色调
        ax4.plot(t, data['omegaz'], label='Z', linewidth=1.5, color='#988ED5')  # 紫色调

        ax1.set_title('position error')

        ax2.set_title('velocity error')
        ax3.set_title('orientation error')
        ax4.set_title('omega error')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        fig.suptitle(f'model_{self.suffix}')
        # plt.show()
        os.makedirs(rf'.\model_set\{self.suffix}\eval_actions.csv')
        if os.path.exists(
                rf'.\model_set\{self.suffix}\eval_actions.csv'):
            os.remove(
                rf'.\model_set\{self.suffix}\eval_actions.csv')

        with open(
                rf'.\model_set\{self.suffix}\eval_actions.csv',
                'w',
                newline='') as f:
            writer = csv.writer(f)
            for item in actions:
                writer.writerow(item)
        f.close()
        plt.savefig(
            rf'.\model_set\{self.suffix}\eval_actions.csv')
        # print(f'model_{seed}.png saved')
        time_list.append((end_time - start_time))


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    return total_kld


seed = np.random.randint(0, 100)
model = train_IL_SRD(hidden_dim=64, seq_len=500, input_dim=13, output_dim=13, nhead=4, num_encoder_layers=3,
                     num_decoder_layers=4, dim_feedforward=256, dropout=0.2, activation='relu', normalize_before=False,
                     return_intermediate_dec=False, ep_num=400, index=50, seed=seed, batch_size=256,
                     lr=7e-4, weight_decay=5e-5)
model.train('mse')
model.eval_new(3000, seed)
