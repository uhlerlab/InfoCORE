import torch
from torch.utils.data import Dataset, Subset
import numpy as np


class DataCollator_Contrast(object):
    """main loader for contrast between drug screens and molecular structure"""
    def __init__(self, replicate, dirmixup, noise, dir_hyp=None, noise_scale=None):
        self.replicate = replicate # replicate=False: use average of replicates as gene expression; dirmixup is deactivated in this case
        self.dirmixup = dirmixup
        self.dir_hyp = dir_hyp
        self.noise = noise
        self.noise_scale = noise_scale

    def __call__(self, batch):
        RG = np.random.default_rng()
        collect_drug = [b['drug'] for b in batch]
        collect_label = [b['label'] for b in batch]
        collect_batches = [b['batch'] for b in batch]
        if not self.replicate:
            diff_ges = np.vstack([np.mean(np.array(b['diff_ges']), axis=0) for b in batch])
            if self.noise:
                diff_ges = diff_ges + self.noise_scale * np.random.randn(diff_ges.size)
            diff_ges = torch.from_numpy(diff_ges).to(torch.float32)
        else:
            diff_list = []
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
                if self.replicate and not self.dirmixup and not self.noise:
                    collect_batches.append(b['batch'][idx])
            diff_ges = torch.from_numpy(np.vstack(diff_list)).to(torch.float32)
        avg_diff_ges = torch.from_numpy(np.vstack([np.mean(np.array(b['diff_ges']), axis=0) for b in batch])).to(torch.float32)
        return {'drug': collect_drug, 'label': collect_label, 'diff_ges': diff_ges, 'avg_diff_ges': avg_diff_ges, 'batch': collect_batches}


class DataCollator_Contrast_NoAug(object):
    """no augmentation analog to DataCollator_Contrast; for validation and test loader"""
    def __init__(self, replicate):
        self.replicate = replicate # replicate=False: use average of replicates as gene expression
        
    def __call__(self, batch):
        collect_drug = [b['drug'] for b in batch]
        collect_label = [b['label'] for b in batch]
        collect_batches = [b['batch'] for b in batch]
        if self.replicate:
            sample_idx = []
            for b in batch:
                sample_idx.append(np.random.choice(np.arange(len(b['diff_ges'])), size=1, replace=False)[0])
            diff_ges = torch.from_numpy(np.vstack([b['diff_ges'][i] for b, i in zip(batch, sample_idx)])).to(torch.float32)
        else:
            diff_ges = torch.from_numpy(np.vstack([np.mean(np.array(b['diff_ges']), axis=0) for b in batch])).to(torch.float32)
        avg_diff_ges = torch.from_numpy(np.vstack([np.mean(np.array(b['diff_ges']), axis=0) for b in batch])).to(torch.float32)
        return {'drug': collect_drug, 'label': collect_label, 'diff_ges': diff_ges, 'avg_diff_ges': avg_diff_ges, 'batch': collect_batches}


class perturbDataset(Dataset):
    """main dataset"""
    def __init__(self, inst_dict):
        self.inst_dict = inst_dict
        self.drugs = []; self.diff_ges = []; self.labels = []; self.batches = []
        self.agg_dict = {} # key: drug id
        for v in self.inst_dict.values():
            did = v[3]; smiles = v[4]; platemap = v[5]; diffges = v[6]
            if did not in self.agg_dict.keys():
                self.agg_dict[did] = {'diff_ges': [diffges], 'batch': [platemap], 'smiles': [smiles]}
            else:
                self.agg_dict[did]['diff_ges'].append(diffges)
                self.agg_dict[did]['batch'].append(platemap)
                self.agg_dict[did]['smiles'].append(smiles)
        for k, v in self.agg_dict.items():
            self.labels.append(k)
            self.drugs.append(v['smiles'][0]); self.batches.append(v['batch'][0])
            self.diff_ges.append(v['diff_ges'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'drug': self.drugs[idx], 'diff_ges': self.diff_ges[idx], 'label': self.labels[idx], 'batch': self.batches[idx]}
        return sample


def dataset2batchsubset(dataset):
    # convert perturbDataset to a dictionary of batch identifier: subset
    batches = [d['batch'] for d in dataset]
    batchidx_dict = {}
    for i, batchid in enumerate(batches):
        if batchid not in batchidx_dict.keys():
            batchidx_dict[batchid] = [i]
        else:
            batchidx_dict[batchid].append(i)
    subset_dict = {b: Subset(dataset, batchidx_dict[b]) for b in batchidx_dict.keys() if len(batchidx_dict[b]) > 10}
    return subset_dict
