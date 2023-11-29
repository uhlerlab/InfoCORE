import pickle
import torch
import argparse
import numpy as np
import os


def load_files():
    data_dir = '/home/wangchy/InfoCORE/GE/data/training_LINCS/'
    with open(os.path.join(data_dir, "inst_info_drug.pkl"), "rb") as fp:
        inst_info_drug = pickle.load(fp)
    with open(os.path.join(data_dir, "ctl_dict1_mean.pkl"), "rb") as fp:
        ctl_dict_mean = pickle.load(fp)
    with open(os.path.join(data_dir, "id_smiles"), "rb") as fp:
        id_smiles = pickle.load(fp)
    with open(os.path.join(data_dir, "mol2vec_mol_repr_dict.pkl"), "rb") as fp:
        drug_repr_dict = pickle.load(fp)
    return inst_info_drug, ctl_dict_mean, id_smiles, drug_repr_dict


def select_inst(inst_info_drug, sel_keys):
    return {k: inst_info_drug[k] for k in sel_keys}


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def selkey_subsetboth_core(inst_info_drug):
    data_dir = '/home/wangchy/InfoCORE/GE/data/training_LINCS/'
    sel_keys_dosesubset = np.load(os.path.join(data_dir, 'sel_keys_dosesubset.npy'))
    corecelllines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP', 'HEPG2']
    sel_keys = []
    for k in sel_keys_dosesubset:
        if k.split('_')[1] in corecelllines:
            sel_keys.append(k)
    print(len(sel_keys))
    return sel_keys


def dict2torch(the_dict):
    return {k: torch.from_numpy(v).float() for k, v in the_dict.items()}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def data_parallel(model):
    if torch.cuda.device_count() >= 1:
        return torch.nn.DataParallel(model)
