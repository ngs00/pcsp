import torch
import numpy
from torch.utils.data import DataLoader
from util.chem import load_elem_attrs
from util.data import load_nist_dataset, collate_fn
from torch_geometric.nn.models import AttentiveFP
from util.model_loader import get_emb_nets
from method.model import Model
from method.periodic_stochastic_process import PeriodicStochProc
from util.cross_modality_retriever import GraphRetriever


dataset_name = 'nist'
random_seed = 0
num_folds = 5
batch_size = 128
window_size = 16
dim_emb = 512
num_epochs = 200
list_acc = list()


# dataset = load_nist_dataset(path_metadata='dataset/nist/metadata.xlsx',
#                             path_jdx='dataset/chemical_analysis/nist/jdx',
#                             idx_smiles=4,
#                             elem_attrs=load_elem_attrs('res/matscholar-embedding.json'))
# torch.save(dataset, 'save/dataset/{}.pt'.format(dataset_name))
dataset = torch.load('save/dataset/{}.pt'.format(dataset_name), weights_only=False)
k_folds = dataset.get_k_folds(num_folds=num_folds, random_seed=random_seed)


for k in range(0, num_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    gt_ids = [d.data_id for d in dataset_test.data]
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_test = DataLoader(dataset_test, batch_size=128, collate_fn=collate_fn)

    emb_net_data, emb_net_struct = get_emb_nets(dataset_name, dataset_train, dim_emb)
    phase_generator = PeriodicStochProc(dim_emb)
    model = Model(emb_net_data, emb_net_struct, phase_generator, dim_emb).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    for epoch in range(0, num_epochs):
        loss_train = model.fit(loader_train, optimizer)
        print('Fold [{}/{}]\tEpoch [{}/{}]\tLoss: {:.3f}'.format(k + 1, num_folds, epoch + 1, num_epochs, loss_train))

        if (epoch + 1) % 50 == 0:
            retriever = GraphRetriever(model)
            retrieval_result = retriever.retrieve(dataset_test, k=1)
            torch.save(model.state_dict(), 'save/{}/model_{}.pt'.format(dataset_name, k))
            print(retrieval_result.calc_acc(gt_ids))

    retriever = GraphRetriever(model)
    retrieval_result = retriever.retrieve(dataset_test, k=1)
    torch.save(model.state_dict(), 'save/{}/model_{}.pt'.format(dataset_name, k))
    list_acc.append(retrieval_result.calc_acc(gt_ids))

print('Top-1 Accuracy: {:.3f} ({:.3f})'.format(numpy.mean(list_acc), numpy.std(list_acc)))
