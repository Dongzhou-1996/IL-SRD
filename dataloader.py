import h5py
import numpy as np
from batchquaterror import BatchQuatOperations
from scipy.optimize import curve_fit
import torch


class Dataloader(object):
    def __init__(self, seq_len):
        super().__init__()
        # self.dir = dir
        self.seq_len = seq_len
        self.batch_q = BatchQuatOperations()

    def prepare_data(self, index, train_ratio, train_batch_size, val_batch_size, seq_len, norm_method):
        shuffled_indices = np.random.permutation(index)
        train_size = int(train_ratio * index)
        train_indices = shuffled_indices[:train_size] + 1
        val_indices = shuffled_indices[train_size:] + 1
        state_dataset = None
        for i in range(1, index + 1):
            with h5py.File(f'.\\dataset_2\\state_data_{i}.h5', 'r') as f:
                data = torch.tensor(f['numeric'][:, :], dtype=torch.float32)
                data = data.unsqueeze(0)

                if state_dataset is None:
                    state_dataset = data
                else:
                    state_dataset = torch.cat((state_dataset, data), dim=0)

        state_dataset = state_dataset.to(torch.device('cuda'))
        noise = torch.normal(mean=0.0, std=0.005, size=state_dataset.shape).to(torch.device('cuda'))
        action_dataset = state_dataset[:, 1:, :].to(torch.device('cuda'))
        action_dataset = torch.cat((action_dataset, state_dataset[:, -1, :].unsqueeze(1)), dim=1)
        incre_dataset = action_dataset[:, 1:, :] - action_dataset[:, :-1, :]
        action_dataset = action_dataset + noise
        action_dataset = action_dataset.to(torch.device('cuda'))

        action_dataset, action_norm_dict = normalization(norm_method, action_dataset, None)
        max_incre = torch.max(torch.max(incre_dataset, dim=0).values, dim=0).values
        max_incre = max_incre.to(torch.device('cuda'))
        min_incre = torch.min(torch.min(incre_dataset, dim=0).values, dim=0).values
        min_incre = min_incre.to(torch.device('cuda'))
        norm = torch.norm(action_dataset[:, :, 6:10], dim=2, keepdim=True)
        action_dataset[:, :, 6:10] = action_dataset[:, :, 6:10] / norm
        state_dataset, state_norm_dict = normalization(norm_method, state_dataset, None)

        train_state_dataset = torch.zeros((train_batch_size, state_dataset.shape[2]), device='cuda')
        train_incre_dataset = torch.zeros((train_batch_size, state_dataset.shape[2]), device='cuda')
        train_padded_action = torch.zeros((train_batch_size, action_dataset.shape[1], action_dataset.shape[2]),
                                          device='cuda')
        train_is_pad = torch.zeros((train_batch_size, action_dataset.shape[1]), device='cuda', dtype=torch.bool)
        train_incre_limit = torch.zeros((train_batch_size, 2, action_dataset.shape[2]), device='cuda', dtype=torch.float32)
        train_indices = train_indices - 1

        for i in range(train_batch_size):
            start_ts = torch.randint(0, state_dataset.shape[1] - 2, (1,)).item()
            episode_idx = torch.randint(0, len(train_indices), (1,)).item()

            train_state_dataset[i, :] = state_dataset[episode_idx, start_ts, :].to('cuda')
            train_incre_dataset[i, :] = incre_dataset[episode_idx, start_ts, :].to('cuda')

            increment = incre_dataset[episode_idx, start_ts: start_ts + seq_len + 1, :]
            batch_max_incre = torch.max(increment, dim=0)[0]
            batch_min_incre = torch.min(increment, dim=0)[0]
            train_incre_limit[i, 0, :] = batch_max_incre.unsqueeze(0)
            train_incre_limit[i, 1, :] = batch_min_incre.unsqueeze(0)
            action = action_dataset[episode_idx, start_ts:, :].to('cuda')
            action_len = action_dataset.shape[1] - start_ts
            train_padded_action[i, :action_len] = action
            train_is_pad[i, action_len:] = True

        train_padded_action = train_padded_action[:, :seq_len, :]
        train_is_pad = train_is_pad[:, :seq_len]

        local_action_max = torch.max(torch.cat((train_padded_action, train_state_dataset.unsqueeze(1)), dim=1), dim=1)[0]
        local_action_min = torch.min(torch.cat((train_padded_action, train_state_dataset.unsqueeze(1)), dim=1), dim=1)[0]

        val_state_dataset = torch.zeros((val_batch_size, state_dataset.shape[2]), device='cuda')
        val_incre_dataset = torch.zeros((val_batch_size, state_dataset.shape[2]), device='cuda')
        val_padded_action = torch.zeros((val_batch_size, action_dataset.shape[1], action_dataset.shape[2]),
                                        device='cuda')
        val_is_pad = torch.zeros((val_batch_size, action_dataset.shape[1]), device='cuda', dtype=torch.bool)

        val_indices = val_indices - 1

        for i in range(val_batch_size):
            start_ts = torch.randint(0, state_dataset.shape[1] - 2, (1,)).item()
            episode_idx = torch.randint(0, len(val_indices), (1,)).item()

            val_state_dataset[i, :] = state_dataset[episode_idx, start_ts, :].to('cuda')
            val_incre_dataset[i, :] = incre_dataset[episode_idx, start_ts, :].to('cuda')

            action = action_dataset[episode_idx, start_ts:, :].to('cuda')
            action_len = action_dataset.shape[1] - start_ts
            val_padded_action[i, :action_len] = action
            val_is_pad[i, action_len:] = True

        val_padded_action = val_padded_action[:, :seq_len, :]
        val_is_pad = val_is_pad[:, :seq_len]
        incre_norm_dict = {
            'max': max_incre,
            'min': min_incre
        }

        # 组织返回的字典
        train_dict = {
            'state': train_state_dataset,
            'incre': train_incre_dataset,
            'action': train_padded_action,
            'is_pad': train_is_pad,
            'incre_limit': train_incre_limit
        }

        val_dict = {
            'state': val_state_dataset,
            'incre': val_incre_dataset,
            'action': val_padded_action,
            'is_pad': val_is_pad
        }

        norm_dict = {
            'state': state_norm_dict,
            'action': action_norm_dict,
            'incre': incre_norm_dict,
            'local_action_max': local_action_max,
            'local_action_min': local_action_min
        }

        return train_dict, val_dict, norm_dict

def normalization(type, dataset, norm_dict=None):
    if not isinstance(dataset, torch.Tensor):
        dataset = torch.tensor(dataset, dtype=torch.float32).to(torch.device('cuda'))

    dims = dataset.ndim
    if norm_dict is None:
        norm_dict = {}
        if type == '-1':
            max_val = torch.max(torch.max(dataset, dim=0)[0], dim=0)[0].to(torch.device('cuda')) if dims == 3 else \
                torch.max(dataset, dim=0)[0]
            min_val = torch.min(torch.min(dataset, dim=0)[0], dim=0)[0].to(torch.device('cuda')) if dims == 3 else \
                torch.min(dataset, dim=0)[0]
            norm_dataset = torch.zeros_like(dataset)

            if dims == 2:
                norm_dataset[:, :6] = 2 * (dataset[:, :6] - min_val[:6]) / (max_val[:6] - min_val[:6]) - 1
                norm_dataset[:, 10:] = 2 * (dataset[:, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:]) - 1
                norm_dataset[:, 6:10] = dataset[:, 6:10]
            elif dims == 3:
                norm_dataset[:, :, :6] = 2 * (dataset[:, :, :6] - min_val[:6]) / (max_val[:6] - min_val[:6]) - 1
                norm_dataset[:, :, 10:] = 2 * (dataset[:, :, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:]) - 1
                norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

            norm_dict = {'max': max_val, 'min': min_val}

        elif type == '0':
            max_val = torch.max(torch.max(dataset, dim=0)[0], dim=0)[0].to(torch.device('cuda')) if dims == 3 else \
                torch.max(dataset, dim=0)[0]
            min_val = torch.min(torch.min(dataset, dim=0)[0], dim=0)[0].to(torch.device('cuda')) if dims == 3 else \
                torch.min(dataset, dim=0)[0]
            norm_dataset = torch.zeros_like(dataset)

            if dims == 2:
                norm_dataset[:, :6] = (dataset[:, :6] - min_val[:6]) / (max_val[:6] - min_val[:6])
                norm_dataset[:, 10:] = (dataset[:, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:])
                norm_dataset[:, 6:10] = dataset[:, 6:10]
            elif dims == 3:
                norm_dataset[:, :, :6] = (dataset[:, :, :6] - min_val[:6]) / (max_val[:6] - min_val[:6])
                norm_dataset[:, :, 10:] = (dataset[:, :, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:])
                norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

            norm_dict = {'max': max_val, 'min': min_val}

        elif type == 'mu':
            mean_val = torch.mean(torch.mean(dataset, dim=0), dim=0).to(
                torch.device('cuda')) if dims == 3 else torch.mean(dataset, dim=0)
            std_val = torch.std(torch.std(dataset, dim=0), dim=0).to(torch.device('cuda')) if dims == 3 else torch.std(
                dataset, dim=0)
            norm_dataset = torch.zeros_like(dataset)

            if dims == 2:
                norm_dataset[:, :6] = (dataset[:, :6] - mean_val[:6]) / std_val[:6]
                norm_dataset[:, 10:] = (dataset[:, 10:] - mean_val[10:]) / std_val[10:]
                norm_dataset[:, 6:10] = dataset[:, 6:10]
            elif dims == 3:
                norm_dataset[:, :, :6] = (dataset[:, :, :6] - mean_val[:6]) / std_val[:6]
                norm_dataset[:, :, 10:] = (dataset[:, :, 10:] - mean_val[10:]) / std_val[10:]
                norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

            norm_dict = {'mean': mean_val, 'std': std_val}

        else:
            print('wrong type')
    else:
        if type == '-1':
            max_val = norm_dict['max']
            min_val = norm_dict['min']
            norm_dataset = torch.zeros_like(dataset)

            if dims == 2:
                norm_dataset[:, :6] = 2 * (dataset[:, :6] - min_val[:6]) / (max_val[:6] - min_val[:6]) - 1
                norm_dataset[:, 10:] = 2 * (dataset[:, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:]) - 1
                norm_dataset[:, 6:10] = dataset[:, 6:10]
            elif dims == 3:
                norm_dataset[:, :, :6] = 2 * (dataset[:, :, :6] - min_val[:6]) / (max_val[:6] - min_val[:6]) - 1
                norm_dataset[:, :, 10:] = 2 * (dataset[:, :, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:]) - 1
                norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

        elif type == '0':
            max_val = norm_dict['max']
            min_val = norm_dict['min']
            norm_dataset = torch.zeros_like(dataset)

            if dims == 2:
                norm_dataset[:, :6] = (dataset[:, :6] - min_val[:6]) / (max_val[:6] - min_val[:6])
                norm_dataset[:, 10:] = (dataset[:, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:])
                norm_dataset[:, 6:10] = dataset[:, 6:10]
            elif dims == 3:
                norm_dataset[:, :, :6] = (dataset[:, :, :6] - min_val[:6]) / (max_val[:6] - min_val[:6])
                norm_dataset[:, :, 10:] = (dataset[:, :, 10:] - min_val[10:]) / (max_val[10:] - min_val[10:])
                norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

        elif type == 'mu':
            mean_val = norm_dict['mean']
            std_val = norm_dict['std']
            norm_dataset = torch.zeros_like(dataset)

            if dims == 2:
                norm_dataset[:, :6] = (dataset[:, :6] - mean_val[:6]) / std_val[:6]
                norm_dataset[:, 10:] = (dataset[:, 10:] - mean_val[10:]) / std_val[10:]
                norm_dataset[:, 6:10] = dataset[:, 6:10]
            elif dims == 3:
                norm_dataset[:, :, :6] = (dataset[:, :, :6] - mean_val[:6]) / std_val[:6]
                norm_dataset[:, :, 10:] = (dataset[:, :, 10:] - mean_val[10:]) / std_val[10:]
                norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

        else:
            print('wrong type')

    return norm_dataset.to(torch.device('cuda')), norm_dict


def denormalization(type, dataset, norm_dict):
    if not isinstance(dataset, torch.Tensor):
        dataset = torch.tensor(dataset, dtype=torch.float32).to(torch.device('cuda'))

    dims = dataset.ndim
    if type == '-1':
        max_val = norm_dict['max']
        min_val = norm_dict['min']
        norm_dataset = torch.zeros_like(dataset)

        if dims == 2:
            norm_dataset[:, :6] = (dataset[:, :6] + 1) * (max_val[:6] - min_val[:6]) / 2 + min_val[:6]
            norm_dataset[:, 10:] = (dataset[:, 10:] + 1) * (max_val[10:] - min_val[10:]) / 2 + min_val[10:]
            norm_dataset[:, 6:10] = dataset[:, 6:10]
        elif dims == 3:
            norm_dataset[:, :, :6] = (dataset[:, :, :6] + 1) * (max_val[:6] - min_val[:6]) / 2 + min_val[:6]
            norm_dataset[:, :, 10:] = (dataset[:, :, 10:] + 1) * (max_val[10:] - min_val[10:]) / 2 + min_val[10:]
            norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

    elif type == '0':
        max_val = norm_dict['max']
        min_val = norm_dict['min']
        norm_dataset = torch.zeros_like(dataset)

        if dims == 2:
            norm_dataset[:, :6] = dataset[:, :6] * (max_val[:6] - min_val[:6]) + min_val[:6]
            norm_dataset[:, 10:] = dataset[:, 10:] * (max_val[10:] - min_val[10:]) + min_val[10:]
            norm_dataset[:, 6:10] = dataset[:, 6:10]
        elif dims == 3:
            norm_dataset[:, :, :6] = dataset[:, :, :6] * (max_val[:6] - min_val[:6]) + min_val[:6]
            norm_dataset[:, :, 10:] = dataset[:, :, 10:] * (max_val[10:] - min_val[10:]) + min_val[10:]
            norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

    elif type == 'mu':
        mean_val = norm_dict['mean']
        std_val = norm_dict['std']
        norm_dataset = torch.zeros_like(dataset)

        if dims == 2:
            norm_dataset[:, :6] = dataset[:, :6] * std_val[:6] + mean_val[:6]
            norm_dataset[:, 10:] = dataset[:, 10:] * std_val[10:] + mean_val[10:]
            norm_dataset[:, 6:10] = dataset[:, 6:10]
        elif dims == 3:
            norm_dataset[:, :, :6] = dataset[:, :, :6] * std_val[:6] + mean_val[:6]
            norm_dataset[:, :, 10:] = dataset[:, :, 10:] * std_val[10:] + mean_val[10:]
            norm_dataset[:, :, 6:10] = dataset[:, :, 6:10]

    else:
        print('wrong type')

    return norm_dataset.to(torch.device('cuda'))
