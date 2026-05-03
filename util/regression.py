import pandas
import torch
import os
from rdkit.Chem import MolFromSmiles
from pymatgen.core import Structure
from util.model_loader import get_emb_nets
from method.periodic_stochastic_process import PeriodicStochProc
from method.model import Model
from util.cross_modality_retriever import GraphRetriever
from util.data import Data, Dataset
from util.chem import get_mol_graph, get_crystal_graph


def load_retriever(dataset_name, dataset, dim_emb):
    emb_net_data, emb_net_struct = get_emb_nets(dataset_name, dataset, dim_emb)
    phase_generator = PeriodicStochProc(dim_emb)
    model = Model(emb_net_data, emb_net_struct, phase_generator, dim_emb).cuda()
    model.load_state_dict(torch.load('save/{}/model_0.pt'.format(dataset_name), map_location=torch.device('cuda:0')))

    return GraphRetriever(model)


def make_joint_dataset(reg_dataset, idx_data_id, idx_target, analytical_dict):
    if reg_dataset.split('.')[-1] == 'xlsx':
        dataset = pandas.read_excel(reg_dataset).values.tolist()
    else:
        dataset = pandas.read_csv(reg_dataset).values.tolist()
    list_data = list()

    for d in dataset:
        if d[idx_data_id] in analytical_dict.keys():
            data = analytical_dict[d[idx_data_id]]
            data.graph.y = torch.tensor(float(d[idx_target]), dtype=torch.float).view(1, 1)
            list_data.append(Data(data.data_id, data.data, data.graph))

    return Dataset(list_data)


def make_real_dataset_mol(path_reg_dataset, elem_attrs, idx_data_id, idx_target, analytical_dataset, retriever):
    analytical_dict = dict()
    for d in analytical_dataset.data:
        analytical_dict[d.data_id] = d
    joint_dataset = make_joint_dataset(path_reg_dataset, idx_data_id, idx_target, analytical_dict)

    list_data = list()
    retrieval_results = retriever.retrieve(joint_dataset, k=1)
    for i in range(0, len(retrieval_results.retrieved_ids)):
        data_id = retrieval_results.retrieved_ids[i][0]

        mol = MolFromSmiles(data_id)
        if mol is None:
            continue

        mol_graph = get_mol_graph(mol, elem_attrs)
        if mol_graph is None:
            continue

        mol_graph.y = joint_dataset.data[i].graph.y
        list_data.append(Data(data_id, joint_dataset.data[i].data, mol_graph))

    return Dataset(list_data)


def make_real_dataset_mat(path_reg_dataset, elem_attrs, idx_data_id, idx_target, analytical_dataset, retriever):
    analytical_dict = dict()
    for d in analytical_dataset.data:
        analytical_dict[d.data_id] = d
    joint_dataset = make_joint_dataset(path_reg_dataset, idx_data_id, idx_target, analytical_dict)

    list_data = list()
    retrieval_results = retriever.retrieve(joint_dataset, k=1)
    for i in range(0, len(retrieval_results.retrieved_ids)):
        data_id = retrieval_results.retrieved_ids[i][0]

        if not os.path.exists('../../data/chem_data/materials_science/calculation/mp/struct/{}.cif'.format(data_id)):
            continue

        mat = Structure.from_file('../../data/chem_data/materials_science/calculation/mp/struct/{}.cif'.format(data_id))
        if mat is None:
            continue

        mat_graph = get_crystal_graph(mat, elem_attrs, atomic_cutoff=4.0)
        if mat_graph is None:
            continue

        mat_graph.y = joint_dataset.data[i].graph.y
        list_data.append(Data(data_id, joint_dataset.data[i].data, mat_graph))

    return Dataset(list_data)


def fit_gnn(model, data_loader, optimizer, loss_func):
    loss_train = 0

    model.train()
    for _, graph in data_loader:
        graph = graph.cuda()
        y = model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        loss = loss_func(y, graph.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.detach().item()

    return loss_train / len(data_loader)


def predict_gnn(model, data_loader):
    preds = list()

    model.eval()
    with torch.no_grad():
        for _, graph in data_loader:
            graph = graph.cuda()
            preds.append(model(graph.x, graph.edge_index, graph.edge_attr, graph.batch))

    return torch.cat(preds, dim=0)
