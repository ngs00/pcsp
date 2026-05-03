import torch
from torch.nn.functional import normalize, cross_entropy


class DataEncoder(torch.nn.Module):
    def __init__(self, emb_net):
        super(DataEncoder, self).__init__()
        self.emb_net = emb_net

    def forward(self, x):
        return normalize(self.emb_net(x), p=2, dim=1)


class GraphEncoder(torch.nn.Module):
    def __init__(self, emb_net):
        super(GraphEncoder, self).__init__()
        self.emb_net = emb_net

    def forward(self, x, edge_index, edge_attr, batch):
        return normalize(self.emb_net(x, edge_index, edge_attr, batch), p=2, dim=1)


class Model(torch.nn.Module):
    def __init__(self, emb_net_data, emb_net_graph, phase_generator, dim_emb=512, inv_tau=14.3):
        super(Model, self).__init__()
        self.data_encoder = DataEncoder(emb_net_data)
        self.graph_encoder = GraphEncoder(emb_net_graph)
        self.phase_generator = phase_generator
        self.dim_emb = dim_emb
        self.inv_tau = torch.tensor(inv_tau, dtype=torch.float)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(2 * dim_emb, dim_emb),
            torch.nn.LeakyReLU()
        )
        self.fc_aggr = torch.nn.Linear(self.phase_generator.num_timesteps, 1)

    def forward(self, data, graph):
        data_emb = self.data_encoder(data)
        graph_emb_init = self.graph_encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        graph_emb_prd = normalize(self.add_dynamics(graph_emb_init), p=2, dim=2)
        graph_emb = self.fc_aggr(graph_emb_prd.swapaxes(1, 2)).squeeze(2)

        return data_emb, normalize(graph_emb, p=2, dim=1)

    def add_dynamics(self, graph_emb):
        z_real, z_img = self.phase_generator(graph_emb)
        z = self.proj(torch.cat([z_real, z_img], dim=2))

        return z

    def fit(self, data_loader, optimizer):
        loss_train = 0

        self.train()
        for data, graph in data_loader:
            data_emb, graph_emb = self(data.cuda(), graph.cuda())
            loss = clip_loss(data_emb, graph_emb, self.inv_tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.detach().item()

        return loss_train / len(data_loader)

    def encode(self, data_loader):
        list_emb1 = list()
        list_emb2 = list()

        with torch.no_grad():
            for d1, d2 in data_loader:
                emb1, emb2 = self(d1.cuda(), d2.cuda())
                list_emb1.append(emb1)
                list_emb2.append(emb2)

        return torch.cat(list_emb1, dim=0), torch.cat(list_emb2, dim=0)

    def calc_sims(self, data_emb, graph_emb):
        return torch.matmul(data_emb, graph_emb.t())


def clip_loss(data_emb, graph_emb, inv_tau):
    logits = inv_tau * torch.matmul(data_emb, graph_emb.t())
    labels = torch.arange(data_emb.shape[0]).cuda()

    return 0.5 * (cross_entropy(logits, labels) + cross_entropy(logits.t(), labels))
