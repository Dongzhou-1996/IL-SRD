import torch
from torch import nn
from torch.autograd import Variable
from backbone import backbone
from transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from positional_embedding import PositionEmbeddingSine
import numpy as np
import math
from dataloader import Dataloader


class ACT(nn.Module):
    def __init__(self, hidden_dim, seq_len, input_dim, output_dim, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout, activation,
                 normalize_before, return_intermediate_dec):
        super().__init__()
        self.num_queries = seq_len
        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionEmbeddingSine(hidden_dim)
        self.backbone = backbone(input_dim, self.hidden_dim)
        self.encoder_action_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder_state_proj = nn.Linear(input_dim, hidden_dim)
        self.input_proj_state = nn.Linear(input_dim, hidden_dim)
        self.latent_dim = 32
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.query_embed = nn.Embedding(seq_len, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.encoder = build_encoder(self.hidden_dim, dropout, nhead, dim_feedforward, num_encoder_layers)
        self.decoder = build_decoder(self.hidden_dim, dropout, nhead, dim_feedforward, num_decoder_layers)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.latent_out_proj = nn.Linear(32, hidden_dim)
        self.input_proj_robot_state = nn.Linear(input_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(1, hidden_dim)
        self.incre_proj = nn.Linear(input_dim, hidden_dim)
        self.local_optimal_proj = nn.Linear(input_dim, hidden_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.type_embedding = nn.Embedding(4, hidden_dim)
        # self.gen_embed = nn.Parameter(torch.randn(self.num_queries, 1, hidden_dim, device=self.device))
        # self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        # self.activation_fn = nn.Tanh()
        # self.max_action = max_action

    def forward(self, states, actions, is_pad=None, local_optimal=None, incre_limit=None, incre_limit_eval=None):
        is_training = actions is not None
        bs, _ = states.shape
        hidden_feature = self.backbone.forward(states.unsqueeze(1))
        # pos_embedding = self.positional_encoding.forward(hidden_feature)  # bs x seq_len x hidden_dim
        # pos_embedding = pos_embedding.unsqueeze(1)
        # pos_embedding = pos_embedding.permute(1, 0, 2)
        # pos_embedding = None
        if is_training:
            action_embed = self.encoder_action_proj(actions)
            # state_embed = self.encoder_state_proj(states)
            # state_embed = state_embed.unsqueeze(1)
            cls_embed = self.cls_embed.weight
            # cls_embed = torch.unsqueeze(cls_embed, 1).repeat(bs, 1, 1)
            encoder_input = torch.cat((hidden_feature, action_embed), dim=1)
            encoder_input = encoder_input.permute(1, 0, 2)
            cls_is_pad = torch.full((bs, 1), False).to(self.device)
            is_pad = torch.concatenate([cls_is_pad, is_pad], axis=1)
            pos_table = get_sinusoid_encoding_table(1+self.num_queries, self.hidden_dim)
            pos_embed = pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_cls = encoder_output[0]

            mu, sigma = encoder_cls.chunk(2, dim=1)
            latent_sample = reparametrize(mu, sigma)
            latent_input = self.latent_out_proj(latent_sample)
            latent_input = latent_input.unsqueeze(0)

            # mu = sigma = None
            # latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(self.device)
            # latent_input = self.latent_out_proj(latent_sample)
            # latent_input = latent_input.unsqueeze(0)
        else:
            mu = sigma = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(self.device)
            latent_input = self.latent_out_proj(latent_sample)
            latent_input = latent_input.unsqueeze(0)
            # local_optimal = torch.tensor([0.5, 1.26, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).unsqueeze(0).to(self.device).to(torch.float32)
            # local_optimal_feat = self.local_optimal_proj(local_optimal)
            # incre_limit_feat = self.incre_proj(incre_limit_eval)

        # proprio_input = self.input_proj_robot_state(states)
        # latent_type = torch.tensor([0], device=self.device).repeat(1, bs)
        # latent_type_embed = self.type_embedding(latent_type)
        # latent_input = latent_input + latent_type_embed

        state_features = self.backbone.forward(states)
        state_features = state_features.unsqueeze(0)

        z_cond = torch.cat([latent_input, state_features], dim=0)
        # gen_carrier = self.gen_embed.repeat(1, bs, 1)
        tgt = state_features.repeat(self.num_queries, 1, 1)
        # tgt = torch.zeros_like(self.query_embed.weight).unsqueeze(1).repeat(1, bs, 1)
        pos_ids = torch.arange(self.num_queries, device=self.device).unsqueeze(0).repeat(bs, 1)
        pos_feats = self.query_embed(pos_ids).transpose(0, 1)
        tgt = tgt + pos_feats
        output = self.decoder(tgt, memory=z_cond, tgt_mask=None)[0]
        pred_actions = self.action_head(output).transpose(0, 1)

        return pred_actions, mu, sigma

    def generate_constraint_mask(self, memory_len):
        """生成memory掩码，增强对约束特征（类型2和3）的注意力权重"""
        # memory_len=5（1+1+2+1），前2个是基础特征（0,1），后3个是约束特征（2,2,3）
        mask = torch.ones((self.num_queries, memory_len))  # 初始全为1（允许关注）
        # 对约束特征（索引2,3,4）赋予更高权重（可选，通过掩码缩放）
        mask[:, 2:] *= 1.5  # 约束特征的注意力权重放大1.5倍
        return mask

def build_encoder(hidden_dim, dropout, nheads, dim_feedforward, num_encoder_layers) :
    d_model = hidden_dim # 256
    dropout = dropout # 0.1
    nhead = nheads # 8
    dim_feedforward = dim_feedforward # 2048
    num_encoder_layers = num_encoder_layers # 4 # TODO shared with VAE decoder
    normalize_before = False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder

def build_decoder(hidden_dim, dropout, nheads, dim_feedforward, num_decoder_layers) :
    d_model = hidden_dim # 256
    dropout = dropout # 0.1
    nhead = nheads # 8
    dim_feedforward = dim_feedforward * 2 # 2048
    num_decoder_layers = num_decoder_layers # 4 # TODO shared with VAE decoder
    normalize_before = False
    activation = "relu"

    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    return decoder



def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0).to(torch.device('cuda'))

# model = ACT(hidden_dim=256, seq_len=20, input_dim=6, output_dim=2, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
#             dim_feedforward=256, dropout=0.1, activation='relu', normalize_before=False,
#             return_intermediate_dec=False).to(torch.device('cuda', 0))
# dataloader = Dataloader('D:\\stable-baselines3-master\\IL\\dataset_full', 100)
# states, actions = dataloader.load_data(50, 10, 8)
# pred_actions = model.forward(states)




