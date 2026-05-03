import numpy
from torch.utils.data import DataLoader
from torch_geometric.nn.models import AttentiveFP
from sklearn.metrics import root_mean_squared_error, r2_score
from tqdm import tqdm
from util.chem import load_elem_attrs
from util.data import collate_fn
from util.regression import *


random_seed = 0
num_repeats = 5
num_folds = 5
dim_emb = 512
num_epochs = 300
analytical_dataset_name = 'nist'
reg_dataset_name = 'esol'
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
analytical_dataset = torch.load('save/dataset/{}.pt'.format(analytical_dataset_name), weights_only=False)
k_folds_analytical = analytical_dataset.get_k_folds(num_folds=num_folds, random_seed=random_seed)
path_reg_dataset = 'dataset/{}.xlsx'.format(reg_dataset_name)


retriever = load_retriever(analytical_dataset_name, analytical_dataset, dim_emb)
dataset_train = make_real_dataset_mol(path_reg_dataset=path_reg_dataset, elem_attrs=elem_attrs,
                                      idx_data_id=0, idx_target=1,
                                      analytical_dataset=k_folds_analytical[0][0], retriever=retriever)
dataset_test = make_real_dataset_mol(path_reg_dataset=path_reg_dataset, elem_attrs=elem_attrs,
                                     idx_data_id=0, idx_target=1,
                                     analytical_dataset=k_folds_analytical[0][1], retriever=retriever)
y_test = torch.tensor([d.graph.y.item() for d in dataset_test.data], dtype=torch.float)
list_rmse = list()
list_r2 = list()

for n in range(0, num_repeats):
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
    loader_test = DataLoader(dataset_test, batch_size=128, collate_fn=collate_fn)
    model = AttentiveFP(in_channels=dataset_train.dim_node_feat,
                        edge_dim=dataset_train.dim_edge_feat,
                        hidden_channels=dim_emb,
                        out_channels=1,
                        num_layers=3,
                        num_timesteps=3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    loss_func = torch.nn.L1Loss()

    for epoch in tqdm(range(0, num_epochs)):
        loss_train = fit_gnn(model, loader_train, optimizer, loss_func)

    preds_test = predict_gnn(model, loader_test).cpu().squeeze(1)
    list_rmse.append(root_mean_squared_error(y_test, preds_test))
    list_r2.append(r2_score(y_test, preds_test))

print('Test MSE: {:.3f} ({:.3f})\tTest R2-score: {:.3f} ({:.3f})'
      .format(numpy.mean(list_rmse), numpy.std(list_rmse), numpy.mean(list_r2), numpy.std(list_r2)))
