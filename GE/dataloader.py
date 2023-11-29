import torch
from torch.utils.data import Dataset, Subset
import numpy as np


class DataCollator_Contrast(object):
    """main loader for contrast between gene expression and molecular structure (+ cell type)"""
    def __init__(self, replicate, dirmixup, noise, dir_hyp=None, noise_scale=None):
        self.replicate = replicate # replicate=False: use average of replicates as gene expression; dirmixup is deactivated in this case
        self.dirmixup = dirmixup
        self.dir_hyp = dir_hyp
        self.noise = noise
        self.noise_scale = noise_scale

    def __call__(self, batch):
        RG = np.random.default_rng()
        collect_celltype = [b['celltype'] for b in batch]
        collect_drug = [b['drug'] for b in batch]
        collect_label = [b['label'] for b in batch]
        if not self.replicate:
            diff_ges = np.vstack([np.mean(np.array(b['diff_ges']), axis=0) for b in batch])
            if self.noise:
                diff_ges = diff_ges + self.noise_scale * np.random.randn(diff_ges.size)
            diff_ges = torch.from_numpy(diff_ges).float()
        else:
            diff_list = []; ctl_list = []; ges_list = []
            for b in batch:
                if len(b['diff_ges']) == 1: # no dirmixup
                    diff_wl = [b['diff_ges'][0]]
                    if self.noise:
                        diff_wl.append(b['diff_ges'][0] + self.noise_scale * np.random.randn(b['diff_ges'][0].size))
                else:
                    diff_wl = [i for i in b['diff_ges']]
                    if self.noise:
                        for g in b['diff_ges']:
                            diff_wl.append(g + self.noise_scale * np.random.randn(g.size))

                    if self.dirmixup:
                        prop = RG.dirichlet((self.dir_hyp,)*len(b['diff_ges']), len(b['diff_ges']))
                        diffstack = np.vstack(b['diff_ges'])
                        mixups_diff = np.matmul(prop, diffstack)
                        for diff in mixups_diff:
                            diff_wl.append(diff)
                idx = np.random.choice(np.arange(len(diff_wl)), size=1)[0]
                diff_list.append(diff_wl[idx])
            diff_ges = torch.from_numpy(np.vstack(diff_list)).float()
        return {'celltype': collect_celltype, 'drug': collect_drug, 'label': collect_label, 'diff_ges': diff_ges}


class DataCollator_Contrast_NoAug(object):
    """no augmentation analog to DataCollator_Contrast"""
    def __init__(self, replicate):
        self.replicate = replicate # replicate=False: use average of replicates as gene expression
        
    def __call__(self, batch):
        collect_celltype = [b['celltype'] for b in batch]
        collect_drug = [b['drug'] for b in batch]
        collect_label = [b['label'] for b in batch]
        if self.replicate:
            sample_idx = []
            for b in batch:
                sample_idx.append(np.random.choice(np.arange(len(b['diff_ges'])), size=1, replace=False)[0])
            diff_ges = torch.from_numpy(np.vstack([b['diff_ges'][i] for b, i in zip(batch, sample_idx)])).float()
        else:
            diff_ges = torch.from_numpy(np.vstack([np.mean(np.array(b['diff_ges']), axis=0) for b in batch])).float()
        return {'celltype': collect_celltype, 'drug': collect_drug, 'label': collect_label, 'diff_ges': diff_ges}


class perturbDataset(Dataset):
    """main dataset"""
    def __init__(self, inst_dict, ctl_dict_mean, id_smiles, drug_repr_dict):
        self.inst_dict = inst_dict
        self.ctl_dict_mean = ctl_dict_mean
        self.id_smiles = id_smiles
        self.celltypes = []; self.drugs = []; self.diff_ges = []; self.labels = []; self.batches = []
        self.agg_dict = {} # key: (cid, did), value: {'diff_ges': [], 'batch': []}
        ctlcelltypes = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP', 'HEPG2']
        for v in self.inst_dict.values():
            batch = v[0]; did = v[2]; cid = v[9]; gexp = v[-1]
            if batch not in self.ctl_dict_mean.keys():
                continue
            if did not in self.id_smiles.keys():
                continue
            if drug_repr_dict is not None and self.id_smiles[did] not in drug_repr_dict.keys():
                continue
            if cid not in ctlcelltypes:
                continue
            diffges = gexp - self.ctl_dict_mean[batch]
            if (cid, did) not in self.agg_dict.keys():
                self.agg_dict[(cid, did)] = {'diff_ges': [diffges], 'batch': [batch]}
            else:
                self.agg_dict[(cid, did)]['diff_ges'].append(diffges)
                self.agg_dict[(cid, did)]['batch'].append(batch)
        for k, v in self.agg_dict.items():
            self.celltypes.append(k[0]); self.labels.append(k[0]+k[1])
            self.drugs.append(self.id_smiles[k[1]])
            self.batches.append(v['batch'])
            self.diff_ges.append(v['diff_ges'])
        
    def __len__(self):
        return len(self.celltypes)

    def __getitem__(self, idx):
        sample = {'celltype': self.celltypes[idx], 'drug': self.drugs[idx], 'diff_ges': self.diff_ges[idx], 'label': self.labels[idx], 'batch': self.batches[idx]}
        return sample


def dataset2cellsubset(dataset):
    # convert perturbDataset to a dictionary of celltype: subset
    cts = [d['celltype'] for d in dataset]
    cellidx_dict = {}
    for i, celltype in enumerate(cts):
        if celltype not in cellidx_dict.keys():
            cellidx_dict[celltype] = [i]
        else:
            cellidx_dict[celltype].append(i)
    subset_dict = {c: Subset(dataset, cellidx_dict[c]) for c in cellidx_dict.keys()}
    return subset_dict


def dataset2batchsubset(dataset):
    # convert perturbDataset to a dictionary of batch identifier: subset
    batches = [d['batch'] for d in dataset]
    batchidx_dict = {}
    for i, batchid in enumerate(batches):
        for b in batchid:
            if b not in batchidx_dict.keys():
                batchidx_dict[b] = [i]
            else:
                batchidx_dict[b].append(i)
    batchidx_dict = {b: list(set(batchidx_dict[b])) for b in batchidx_dict.keys() if len(set(batchidx_dict[b])) > 10}
    subset_dict = {b: Subset(dataset, batchidx_dict[b]) for b in batchidx_dict.keys()}
    return subset_dict


def new_drug_sampler(drugs, seed=0):
    # train-val-test split based on drug
    drugs_idx_dict = {}
    for i, d in enumerate(drugs):
        try:
            drugs_idx_dict[d].append(i)
        except KeyError:
            drugs_idx_dict[d] = [i]
    np.random.seed(seed)
    random_dorder = np.random.permutation(list(drugs_idx_dict.keys()))
    train_idx = []
    val_idx = []
    test_idx = []
    for d in random_dorder:
        if len(train_idx) < int(len(drugs)*0.8):
            train_idx.extend(drugs_idx_dict[d])
        elif len(val_idx) < int(len(drugs)*0.1):
            val_idx.extend(drugs_idx_dict[d])
        else:
            test_idx.extend(drugs_idx_dict[d])
    return train_idx, val_idx, test_idx
