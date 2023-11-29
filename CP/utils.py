import pickle
import torch
import argparse
import numpy as np
import os


def load_files():
    data_dir = '/home/wangchy/InfoCORE/CP/data/training_CDRP_Bray2017/'
    with open(os.path.join(data_dir, "inst_info_drug_cp.pkl"), "rb") as fp:
        # drug screens data
        inst_info_drug = pickle.load(fp)
    with open(os.path.join(data_dir, "mol2vec_mol_repr_dict_cp.pkl"), "rb") as fp:
        # drug structures featurized by Mol2Vec embedding
        drug_repr_dict = pickle.load(fp)
    return inst_info_drug, drug_repr_dict


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
