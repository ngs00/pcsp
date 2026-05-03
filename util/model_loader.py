from torch_geometric.nn.models import AttentiveFP
from torchvision.models import resnet34
from ml.nn import SeqConvNN, CGCNN


def get_emb_nets(dataset_name, dataset, dim_emb, window_size=16):
    if dataset_name == 'qme14s_ir':
        emb_net_data = SeqConvNN(dim_emb=dim_emb,
                                 len_seq=dataset.data[0].data.shape[0],
                                 window_size=window_size)
        emb_net_struct = AttentiveFP(in_channels=dataset.dim_node_feat,
                                     edge_dim=dataset.dim_edge_feat,
                                     hidden_channels=dim_emb,
                                     out_channels=dim_emb,
                                     num_layers=3,
                                     num_timesteps=3)
    elif dataset_name == 'qme14s_raman':
        emb_net_data = SeqConvNN(dim_emb=dim_emb,
                                 len_seq=dataset.data[0].data.shape[0],
                                 window_size=window_size)
        emb_net_struct = AttentiveFP(in_channels=dataset.dim_node_feat,
                                     edge_dim=dataset.dim_edge_feat,
                                     hidden_channels=dim_emb,
                                     out_channels=dim_emb,
                                     num_layers=3,
                                     num_timesteps=3)
    elif dataset_name == 'open_xrd':
        emb_net_data = SeqConvNN(dim_emb=dim_emb, len_seq=dataset.data[0].data.shape[0],
                                 len_seq_emb=364,
                                 window_size=window_size)
        emb_net_struct = CGCNN(dim_node_feat=dataset.dim_node_feat,
                               dim_edge_feat=dataset.dim_edge_feat,
                               dim_latent=dim_emb,
                               dim_out=dim_emb)
    elif dataset_name == 'ins_mat':
        emb_net_data = resnet34(num_classes=dim_emb)
        emb_net_struct = CGCNN(dim_node_feat=dataset.dim_node_feat,
                               dim_edge_feat=dataset.dim_edge_feat,
                               dim_latent=dim_emb,
                               dim_out=dim_emb)
    elif dataset_name == 'nist':
        emb_net_data = SeqConvNN(dim_emb=dim_emb,
                                 len_seq=dataset.data[0].data.shape[0],
                                 len_seq_emb=396,
                                 window_size=window_size)
        emb_net_struct = AttentiveFP(in_channels=dataset.dim_node_feat,
                                     edge_dim=dataset.dim_edge_feat,
                                     hidden_channels=dim_emb,
                                     out_channels=dim_emb,
                                     num_layers=3,
                                     num_timesteps=3)
    elif dataset_name == 'mol_nmr2d':
        emb_net_data = resnet34(num_classes=dim_emb)
        emb_net_struct = AttentiveFP(in_channels=dataset.dim_node_feat,
                                     edge_dim=dataset.dim_edge_feat,
                                     hidden_channels=dim_emb,
                                     out_channels=dim_emb,
                                     num_layers=3,
                                     num_timesteps=3)
    else:
        raise KeyError

    return emb_net_data, emb_net_struct
