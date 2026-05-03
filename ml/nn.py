import torch
from torch.nn.functional import leaky_relu, normalize
from torch_geometric.nn.conv import CGConv
from torch_geometric.nn.glob import global_mean_pool


class SeqConvNN(torch.nn.Module):
    def __init__(self, dim_emb, len_seq, len_seq_emb=427, window_size=16):
        super(SeqConvNN, self).__init__()
        self.len_seq = len_seq
        self.len_window_seq = len_seq - window_size + 1
        self.window_size = window_size

        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=window_size, stride=2)
        self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=window_size, stride=2)
        self.fc_feat = torch.nn.Linear(16, 1)
        self.fc_seq = torch.nn.Sequential(
            torch.nn.Linear(len_seq_emb, dim_emb),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dim_emb, dim_emb)
        )

    def forward(self, spect):
        h = leaky_relu(self.conv1(spect.unsqueeze(2).swapaxes(1, 2)))
        h = leaky_relu(self.conv2(h))
        h = leaky_relu(self.fc_feat(h.swapaxes(1, 2))).squeeze(2)
        out = self.fc_seq(h)

        return out


class CGCNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_latent, dim_out):
        super(CGCNN, self).__init__()
        self.fc_atom = torch.nn.Linear(dim_node_feat, dim_latent)
        self.act_fc_atom = torch.nn.PReLU()
        self.gc1 = CGConv(dim_latent, dim_edge_feat)
        self.act_gc1 = torch.nn.PReLU()
        self.gc2 = CGConv(dim_latent, dim_edge_feat)
        self.act_gc2 = torch.nn.PReLU()
        self.fc_lin = torch.nn.Linear(dim_latent, dim_out)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.act_fc_atom(self.fc_atom(x))
        h = self.act_gc1(self.gc1(h, edge_index, edge_attr))
        h = self.act_gc2(self.gc2(h, edge_index, edge_attr))
        hg = normalize(global_mean_pool(h, batch), p=2, dim=1)
        out = self.fc_lin(hg)

        return out
