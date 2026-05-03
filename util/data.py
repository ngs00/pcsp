import pandas
import warnings
import jcamp
import glob
import os
from itertools import chain
from tqdm import tqdm
from scipy import interpolate
from scipy.signal import savgol_filter
from rdkit.Chem import MolFromSmiles
from PIL import Image
from pymatgen.core import Structure, Lattice
from ast import literal_eval
from torch_geometric.data import Batch
from torchvision.transforms import ToTensor
from util.chem import *


warnings.filterwarnings(action='ignore')


class Data:
    def __init__(self, data_id, data, graph):
        self.data_id = data_id
        self.data = data
        self.graph = graph


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.data_ids = [d.data_id for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].data, self.data[idx].graph

    @property
    def dim_node_feat(self):
        return self.data[0].graph.x.shape[1]

    @property
    def dim_edge_feat(self):
        return self.data[0].graph.edge_attr.shape[1]

    def get_k_folds(self, num_folds, random_seed=None):
        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.array_split(numpy.random.permutation(len(self.data)), num_folds)
        sub_datasets = list()
        for i in range(0, num_folds):
            sub_datasets.append([self.data[idx] for idx in idx_rand[i]])

        k_folds = list()
        for i in range(0, num_folds):
            dataset_train = Dataset(list(chain.from_iterable(sub_datasets[:i] + sub_datasets[i + 1:])))
            dataset_test = Dataset(sub_datasets[i])
            k_folds.append([dataset_train, dataset_test])

        return k_folds


def load_nist_dataset(path_metadata, path_jdx, idx_smiles, elem_attrs):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    data = list()

    for i in tqdm(range(0, len(metadata))):
        irs = read_jdx_file(path_jdx + '/{}.jdx'.format(metadata[i][0]), norm_y=True, wmin=550, wmax=3801)

        if irs is None:
            continue

        if pandas.isnull(metadata[i][idx_smiles]):
            continue

        if get_state_label(metadata[i][3]) != 'gas':
            continue

        mol = MolFromSmiles(metadata[i][idx_smiles])
        if mol is None:
            continue

        mol_graph = get_mol_graph(mol, elem_attrs)
        if mol_graph is None:
            continue

        data.append(Data(metadata[i][idx_smiles], irs, mol_graph))

    return Dataset(data)


def load_qme14s_ir_dataset(path_data, elem_attrs):
    files = glob.glob(path_data + '/*.csv')
    data = list()

    for i in tqdm(range(0, len(files))):
        with open(files[i], 'r') as f:
            d = f.readlines()

        smiles = d[0].replace('\n', '')
        mol = MolFromSmiles(smiles)
        if mol is None:
            continue

        mol_graph = get_mol_graph(mol, elem_attrs)
        if mol_graph is None:
            continue

        absorbance = None
        for n in range(0, len(d)):
            if d[n] == '\n':
                absorbance = numpy.array([float(v) for v in d[n + 1].split(',')])
                break
        absorbance = numpy.nan_to_num(absorbance, nan=0)
        absorbance = savgol_filter(absorbance, 32, 3)[::2]
        absorbance = (absorbance - numpy.min(absorbance)) / (numpy.max(absorbance) - numpy.min(absorbance))
        absorbance = torch.tensor(absorbance, dtype=torch.float).nan_to_num(nan=0)

        if torch.isnan(absorbance).any():
            print(smiles)

        data.append(Data(smiles, absorbance, mol_graph))

    return Dataset(data)


def load_qme14s_dataset(path_data, elem_attrs):
    files = glob.glob(path_data + '/*.csv')
    data = list()

    for i in tqdm(range(0, len(files))):
        with open(files[i], 'r') as f:
            d = f.readlines()

        smiles = d[0].replace('\n', '')
        mol = MolFromSmiles(smiles)
        if mol is None:
            continue

        mol_graph = get_mol_graph(mol, elem_attrs)
        if mol_graph is None:
            continue

        absorbance = None
        for n in range(0, len(d)):
            if d[n] == '\n':
                absorbance = numpy.array([float(v) for v in d[n + 1].split(',')])
                break
        absorbance = numpy.nan_to_num(absorbance, nan=0)
        absorbance = savgol_filter(absorbance, 32, 3)[::2]
        absorbance = (absorbance - numpy.min(absorbance)) / (numpy.max(absorbance) - numpy.min(absorbance))
        absorbance = torch.tensor(absorbance, dtype=torch.float).nan_to_num(nan=0)

        data.append(Data(smiles, absorbance, mol_graph))

    return Dataset(data)


def load_ins_dataset(path_metadata, path_data, idx_mp_id, elem_attrs):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    img_transform = ToTensor()
    data = list()

    for i in tqdm(range(0, len(metadata))):
        img = Image.open(path_data + '/{}/spectrum2d.png'.format(metadata[i][idx_mp_id])).convert('RGB')
        img = img_transform(numpy.array(img))

        mat = Structure.from_file(path_data + '/{}/structure.cif'.format(metadata[i][idx_mp_id]))
        if mat is None:
            continue

        mat_graph = get_crystal_graph(mat, elem_attrs, atomic_cutoff=3.0)
        if mat_graph is None:
            continue

        data.append(Data(metadata[i][idx_mp_id], img, mat_graph))

    return Dataset(data)


def read_jdx_file(file_name, norm_y, wmin=None, wmax=None):
    spect = jcamp.jcamp_readfile(file_name)

    if spect['yunits'] != 'ABSORBANCE' and spect['yunits'] != 'TRANSMITTANCE':
        return None

    if 'path length' in spect.keys():
        del spect['path length']
    jcamp.jcamp_calc_xsec(spect, skip_nonquant=False)

    if spect['yunits'] == 'ABSORBANCE':
        spect['absorbance'] = spect['y']

    if numpy.min(spect['wavenumbers']) > 1000:
        return None
    if numpy.max(spect['wavenumbers']) < 3000:
        return None

    spect['absorbance'] = numpy.nan_to_num(spect['absorbance'], nan=0)
    spect['wavenumbers'], spect['absorbance'] = interpol_absorbance(spect['wavenumbers'],
                                                                    spect['absorbance'],
                                                                    wmin,
                                                                    wmax)
    if norm_y:
        spect['absorbance'] = spect['absorbance'] / numpy.max(spect['absorbance'])

    absorbance = torch.tensor(spect['absorbance'], dtype=torch.float)
    absorbance_savgol = torch.tensor(savgol_filter(absorbance, 32, 3), dtype=torch.float)

    return absorbance_savgol


def interpol_absorbance(wavenumber, absorbance, wmin, wmax):
    f_interpol = interpolate.interp1d(wavenumber, absorbance, kind='linear', fill_value='extrapolate')

    if wmin is None or wmax is None:
        _wavenumber = numpy.arange(int(numpy.min(wavenumber)), int(numpy.max(wavenumber)), step=2)
    else:
        _wavenumber = numpy.arange(int(wmin), int(wmax), step=2)
    _absorbance = f_interpol(_wavenumber)
    _absorbance = _absorbance.clip(min=0, max=1)

    return _wavenumber, _absorbance


def load_xrd_dataset(path_data, elem_attrs):
    files = glob.glob(path_data + '/*.json')
    data = list()

    for fname in tqdm(files):
        with open(fname, 'r') as f:
            json_data = json.load(f)

        _, seq = interpol_xrd(json_data['two_theta_values'], json_data['intensities'])
        seq = torch.tensor(seq, dtype=torch.float)

        struct_info = json.loads(json_data['label'])
        phase = json.loads(struct_info['phases'][0])
        if phase['basis'] is None:
            continue

        lattice_params = literal_eval(phase['lattice'])
        lattice = Lattice.from_parameters(a=lattice_params[0], b=lattice_params[1], c=lattice_params[2],
                                          alpha=lattice_params[3], beta=lattice_params[4], gamma=lattice_params[5])
        species = list()
        coords = list()

        basis = json.loads(phase['basis'])
        if len(basis) > 1000:
            continue

        for b in basis:
            _b = json.loads(b)
            species.append(_b['symbol'].replace('0+', ''))
            coords.append([_b['x'], _b['y'], _b['z']])

        mat = Structure(lattice, species, coords)

        mat_graph = get_crystal_graph(mat, elem_attrs)
        if mat_graph is None:
            continue

        data_id = fname.split('/')[-1].split('.')[0]
        data.append(Data(data_id, seq, mat_graph))

    return Dataset(data)


def interpol_xrd(theta, intensity):
    f_interpol = interpolate.interp1d(theta, intensity,
                                      kind='linear',
                                      bounds_error=False,
                                      fill_value=(0, 0))
    _theta = numpy.arange(0, 150, step=0.1)
    _intensity = f_interpol(_theta)
    _intensity = (_intensity - numpy.min(_intensity)) / (numpy.max(_intensity) - numpy.min(_intensity))

    return _theta, _intensity


def load_cell_painting_dataset(path_metadata, path_data, idx_data_id, idx_smiles, elem_attrs):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    img_transform = ToTensor()
    data = list()

    for i in tqdm(range(0, len(metadata))):
        img = Image.open(path_data + '/{}.png'.format(metadata[i][idx_data_id])).convert('RGB')
        img = img_transform(numpy.array(img))

        mol = MolFromSmiles(metadata[i][idx_smiles])
        if mol is None:
            continue

        mol_graph = get_mol_graph(mol, elem_attrs)
        if mol_graph is None:
            continue

        data.append(Data(metadata[i][idx_data_id], img, mol_graph))

    return Dataset(data)


def load_mol_nmr_dataset(path_metadata, path_data, idx_filename):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    img_transform = ToTensor()
    data = list()

    for i in tqdm(range(0, len(metadata))):
        if not os.path.isdir(path_data + '/{}'.format(metadata[i][idx_filename])):
            continue

        img = Image.open(path_data + '/{}/nmr2d.png'.format(metadata[i][idx_filename])).convert('RGB')
        img = img_transform(numpy.array(img))
        mol_graph = torch.load(path_data + '/{}/mol_graph.pt'.format(metadata[i][idx_filename]), weights_only=False)
        data.append(Data(metadata[i][idx_filename], img, mol_graph))

    return Dataset(data)


def collate_fn(batch):
    list_data = list()
    list_graphs = list()

    for b in batch:
        list_data.append(b[0].unsqueeze(0))
        list_graphs.append(b[1])

    return torch.cat(list_data, dim=0), Batch.from_data_list(list_graphs)
