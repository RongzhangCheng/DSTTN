import torch
import torch.nn as nn
from net_component import Transformer


class SpatialTemporalEmbeddingLayer(nn.Module):

    def __init__(self, edim=32, num_nodes=1, input_dim=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = edim
        self.input_len = 12
        self.input_dim = input_dim
        self.embed_dim = edim
        self.temp_dim_tid = edim
        self.temp_dim_diw = edim

        self.if_time_in_day = True
        self.if_day_in_week = True
        self.if_spatial = True

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(288, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        input_data = x[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = x[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]

        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = x[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # spatial embeddings
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        if time_in_day_emb is not None:

            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        h = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        return h


class TransformerLayer(nn.Module):
    def __init__(self, in_dim, layers=1, dropout=.1, heads=8, num_nodes=1, batch_size=32):
        super().__init__()
        self.transformer = Transformer.Transformer(in_dim, heads, layers, layers, in_dim * 4, dropout=dropout,
                                                   num_nodes=num_nodes, batch_size=batch_size)

    def forward(self, input):
        x = input.permute(1, 0, 2)
        x = self.transformer(x, x)
        return x.permute(1, 0, 2)


class network(nn.Module):
    def __init__(self, dropout=0.1, edim=32, out_dim=12, hid_dim=64, layers=6, batch_size=32, num_nodes=1, input_dim=3):
        super(network, self).__init__()
        self.stelayer = SpatialTemporalEmbeddingLayer(edim=edim, num_nodes=num_nodes, input_dim=input_dim)
        self.conv = nn.Conv2d(in_channels=edim * 4, out_channels=hid_dim, kernel_size=(1, 1))
        self.translyear = TransformerLayer(in_dim=hid_dim, layers=layers, dropout=dropout, num_nodes=num_nodes,
                                           batch_size=batch_size)
        self.lin = nn.Linear(hid_dim, out_dim)

    def forward(self, input):
        x = input.transpose(1, 3)
        x = self.stelayer(x)
        x = self.conv(x)[..., -1]
        x = x.transpose(1, 2)
        x = self.translyear(x)
        x = self.lin(x)
        return x.transpose(1, 2).unsqueeze(-1)
