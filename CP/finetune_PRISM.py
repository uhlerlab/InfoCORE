from torch.utils.data import DataLoader, Subset, Dataset
import torch
from torch import nn
from utils import * 
from metrics import *
from argparse import ArgumentParser
import numpy as np
from model import MoCoPair, SimCLRPair
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
import copy
import deepchem as dc
from ordered_set import OrderedSet
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    parser = ArgumentParser()
    ## model params
    parser.add_argument('--model', type=str, default='MoCo') # MoCo or SimCLR
    parser.add_argument('--classify_intldim', type=int, default=512) # 0 if linear classifier
    parser.add_argument('--infocore', type=str2bool, default=False)
    parser.add_argument('--mi_weight_coef', type=float, default=1.0)
    parser.add_argument('--anchor_logit', type=str, default='q') # q or k
    parser.add_argument('--rewgrad_coef', type=float, default=0.0) # weight of adv gradient
    parser.add_argument('--logitdim', type=int, default=97) # number of categorical batches
    parser.add_argument('--proj', type=str, default='none') # none, linear, mlp
    parser.add_argument('--dataparallel', type=str2bool, default=False)

    ## MLP params
    # drug screen embedding (X_g, Z_g)
    parser.add_argument('--gene_dim', type=int, default=701) # input data dimension
    parser.add_argument('--enc_nlayers', type=int, default=3)
    parser.add_argument('--enc_intldim', type=int, default=768)
    parser.add_argument('--enc_hiddim', type=int, default=256)
    # drug structure embedding (X_d, Z_d)
    parser.add_argument('--dc_nlayers', type=int, default=3)
    parser.add_argument('--dc_intldim', type=int, default=2560)

    ## loss params
    parser.add_argument('--temperature', type=float, default=0.1)

    ## MoCo params
    parser.add_argument('--moco_m', type=float, default=0.999)
    parser.add_argument('--moco_m_clf', type=float, default=0.999)
    parser.add_argument('--moco_k', type=int, default=4096)

    ## finetuning params
    parser.add_argument('--ft_lr', type=float, default=1e-4)
    parser.add_argument('--ft_n_epoches', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_dir', type=str) 
    parser.add_argument('--criterion', type=str, default='r2') 
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    inst_info_drug, drug_repr_dict = load_files()
    sel_keys = list(inst_info_drug.keys())
    inst_info_drug = select_inst(inst_info_drug, sel_keys)
    
    scaler_drug = StandardScaler(mean=torch.from_numpy(np.mean(np.array(list(drug_repr_dict.values())), 0)),
                                std=torch.from_numpy(np.std(np.array(list(drug_repr_dict.values())), 0)))
    drug_repr_dict = {k: torch.clip(scaler_drug.transform(v), -2.5, 2.5) for k, v in dict2torch(drug_repr_dict).items()} 
    
    # load PRISM data
    with open("/home/wangchy/InfoCORE/CP/data/PRISM/depmap_dict.pkl", 'rb') as fp:
        depmap_dict = pickle.load(fp)
    with open("/home/wangchy/InfoCORE/CP/data/PRISM/id_smiles", 'rb') as fp:
        id_smiles = pickle.load(fp)
    druglist = list(set(id_smiles.keys()) & set([k[-13:] for k in depmap_dict.keys()]))
    druglist = [id_smiles[k] for k in druglist]

    cmap_depmap_dict = {k: depmap_dict[k] for k in depmap_dict.keys() if k[:-13] in ['A549', 'MCF7', 'A375', 'PC3', 'HT29']}
    nmin, nmax = np.quantile(list(cmap_depmap_dict.values()), [0.15, 0.85])
    depmap_dict = {k: v for k, v in depmap_dict.items() if v < nmin or v > nmax}
    
    featurizer = dc.feat.Mol2VecFingerprint()
    features = featurizer.featurize(druglist)
    mol2vec_eval = {k: features[i] for k, i in zip(druglist, range(features.shape[0]))}
    mol2vec_eval = {k: torch.clip(scaler_drug.transform(v), -2.5, 2.5) for k, v in dict2torch(mol2vec_eval).items()} 
    drug_repr_dict.update(mol2vec_eval)
     
    if args.model == 'MoCo':
        model = MoCoPair(args, drug_repr_dict, device).to(device)
    elif args.model == 'SimCLR':
        model = SimCLRPair(args, drug_repr_dict, device).to(device)

    if args.dataparallel:
        model = data_parallel(model)

    print(args.model_dir)
    if args.model_dir != 'rdm':
        save_dir_sub = [f for f in os.listdir('/home/wangchy/InfoCORE/CP/model_save/') if f.endswith(args.model_dir)][0]
        save_dir = os.path.join('/home/wangchy/InfoCORE/CP/model_save/', save_dir_sub)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar'), map_location=device))
        if args.dataparallel:
            model = model.module

    class finetune(nn.Module):
        def __init__(self, model):
            super(finetune, self).__init__()
            self.model = model
            self.fc = nn.Linear(args.enc_hiddim, 1)

        def forward(self, d):
            if args.model == 'MoCo':
                demb = self.model.encoder_q.get_dc_embedding(d)
            elif args.model == 'SimCLR':
                demb = self.model.encoder.get_dc_embedding(d)
            return self.fc(demb)
    
    class ftdataset(Dataset):
        def __init__(self, drugs_y_dict):
            self.drugs = list(drugs_y_dict.keys())
            self.y = np.vstack([drugs_y_dict[k] for k in self.drugs])
        def __len__(self):
            return len(self.drugs)
        def __getitem__(self, idx):
            return {'drug': self.drugs[idx], 'label': torch.from_numpy(self.y[idx]).to(torch.float32).to(device)}


    cnames = ['A549', 'MCF7', 'A375', 'PC3', 'HT29']

    def scaffold_splitting():
        druglist = list(OrderedSet(id_smiles.keys()) & OrderedSet([k[-13:] for k in depmap_dict.keys()]))
        druglist = [id_smiles[k] for k in druglist]
        # avoid having drugs in pretraining to be in the validation set of finetuning
        train_smiles = np.load('/home/wangchy/InfoCORE/CP/data/training_CDRP_Bray2017/cp_train_smiles.npy')
        whole_drugs_y_dict = {cname+k: depmap_dict[cname+k] for k in id_smiles.keys() for cname in ['A549', 'MCF7', 'A375', 'PC3', 'HT29'] 
                      if id_smiles[k] in druglist and cname+k in depmap_dict.keys()}
        wholedids = list(OrderedSet([k[-13:] for k in whole_drugs_y_dict.keys()]))
        wholesmiles = [id_smiles[k] for k in wholedids]
        train_smiles_depmap = list(OrderedSet(train_smiles) & OrderedSet(wholesmiles))
        scaffolds = list(OrderedSet([MurckoScaffold.MurckoScaffoldSmiles(smiles) for smiles in wholesmiles]))
        np.random.seed(1)
        mocotrain_scaffolds = list(OrderedSet([MurckoScaffold.MurckoScaffoldSmiles(smiles) for smiles in train_smiles_depmap]))
        left_scaffolds = [s for s in scaffolds if s not in mocotrain_scaffolds]
        shuffleidx = np.random.permutation(len(left_scaffolds))
        train_list = []; val_list = []; test_list = []
        shuffleidx_list = [shuffleidx[int(a*len(shuffleidx)): int((a+0.1)*len(shuffleidx))] for a in np.arange(0, 1, 0.1)]
        trcvselect = [[2,3,4,5,6,7,8,9], [3,4,5,6,7,8,9,0], [4,5,6,7,8,9,0,1], [5,6,7,8,9,0,1,2], [6,7,8,9,0,1,2,3], [7,8,9,0,1,2,3,4], [8,9,0,1,2,3,4,5], [9,0,1,2,3,4,5,6], [0,1,2,3,4,5,6,7], [1,2,3,4,5,6,7,8]]
        for i in range(10):
            train_scaffolds = mocotrain_scaffolds + [left_scaffolds[j] for j in [w for v in [shuffleidx_list[k] for k in trcvselect[i]] for w in v]]
            val_scaffolds = [left_scaffolds[j] for j in shuffleidx_list[i]]
            test_scaffolds = [left_scaffolds[j] for j in shuffleidx_list[(i+1)%10]]

            train_dids_depmap = [k for k in wholedids if MurckoScaffold.MurckoScaffoldSmiles(id_smiles[k]) in train_scaffolds]
            val_dids_depmap = [k for k in wholedids if MurckoScaffold.MurckoScaffoldSmiles(id_smiles[k]) in val_scaffolds]
            test_dids_depmap = [k for k in wholedids if MurckoScaffold.MurckoScaffoldSmiles(id_smiles[k]) in test_scaffolds]
            train_list.append(train_dids_depmap)
            val_list.append(val_dids_depmap)
            test_list.append(test_dids_depmap)

        return {'train': train_list, 'val': val_list, 'test': test_list}
    
    trainvaltestdids = scaffold_splitting()
    
    preds = []; labels = []
    preds_val = []; labels_val = []
    for cname in cnames:
        whole_drugs_y_dict = {id_smiles[k]: depmap_dict[cname+k] for k in id_smiles.keys() if id_smiles[k] in druglist and cname+k in depmap_dict.keys()}
        preds_c = []; labels_c = []
        preds_c_val = []; labels_c_val = []
        for a in range(10):
            train_drugs_final = list(set([id_smiles[k] for k in trainvaltestdids['train'][a]]) & set(whole_drugs_y_dict.keys()))
            val_drugs = list(set([id_smiles[k] for k in trainvaltestdids['val'][a]]) & set(whole_drugs_y_dict.keys()))
            test_drugs = list(set([id_smiles[k] for k in trainvaltestdids['test'][a]]) & set(whole_drugs_y_dict.keys()))

            trainft_dataset = ftdataset({k: whole_drugs_y_dict[k] for k in train_drugs_final})
            valft_dataset = ftdataset({k: whole_drugs_y_dict[k] for k in val_drugs})
            testft_dataset = ftdataset({k: whole_drugs_y_dict[k] for k in test_drugs})
            trainft_loader = DataLoader(trainft_dataset, batch_size=args.batch_size, shuffle=True)
            valft_loader = DataLoader(valft_dataset, batch_size=args.batch_size, shuffle=False)
            testft_loader = DataLoader(testft_dataset, batch_size=args.batch_size, shuffle=False)
            trainval_dataset = ftdataset({k: whole_drugs_y_dict[k] for k in train_drugs_final+val_drugs})
            
            def smi2scaffold(smi):
                try:
                    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=True)
                except:
                    return smi
            splitter = GroupKFold(n_splits=5)
            splitted = splitter.split(trainval_dataset, trainval_dataset.y, [smi2scaffold(d) for d in trainval_dataset.drugs])
            
            train_list = []; val_list = []
            for train_idx, val_idx in splitted:
                train_list.append(DataLoader(Subset(trainval_dataset, train_idx), batch_size=args.batch_size, shuffle=True))
                val_list.append(DataLoader(Subset(trainval_dataset, val_idx), batch_size=args.batch_size, shuffle=False))

            criterion = nn.MSELoss()

            best_model_list = []
            for nfold in range(5):
                # 5-fold CV in train-val set
                trainft_loader = train_list[nfold]
                valft_loader = val_list[nfold]
                ftmodel = finetune(copy.deepcopy(model)).to(device)
                optimizer_ft = torch.optim.Adam(ftmodel.parameters(), lr=args.ft_lr)
                lr_scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1, last_epoch=-1, verbose=False)
                best_loss = 1e10
                best_val_mse = 1e10
                best_val_r2 = -1e10
                for epoch in range(args.ft_n_epoches):
                    running_loss = 0.0
                    ftmodel.train()
                    for i, batch in enumerate(trainft_loader):
                        optimizer_ft.zero_grad()
                        outputs = ftmodel(batch['drug'])
                        loss = criterion(outputs, batch['label'].to(device))
                        loss.backward()
                        optimizer_ft.step()
                        running_loss += loss.item()
                    lr_scheduler_ft.step()
                    running_loss /= (i+1)
                    ftmodel.eval(); 
                    with torch.no_grad():
                        val_loss = 0.0
                        val_preds = []; val_labels = []
                        for i, batch in enumerate(valft_loader):
                            outputs = ftmodel(batch['drug'])
                            loss = criterion(outputs, batch['label'].to(device))
                            val_loss += loss.item()
                            val_preds.append(outputs.detach().cpu().numpy())
                            val_labels.append(batch['label'].detach().cpu().numpy())
                        val_r2 = r2_score(np.vstack(val_labels), np.vstack(val_preds))
                        val_mse = mean_squared_error(np.vstack(val_labels), np.vstack(val_preds))
                        val_loss /= (i+1)
                        if ((args.criterion=='r2' and val_r2 > best_val_r2) or (args.criterion=='mse' and val_mse < best_val_mse)):
                            best_val_mse = val_mse
                            best_loss = val_loss
                            best_val_r2 = val_r2
                            best_epoch = epoch
                            best_ftmodel = copy.deepcopy(ftmodel)
                best_model_list.append(best_ftmodel)

            test_preds_all = []; val_preds_all = []; val_labels_all = []
            for nfold in range(5):
                ftmodel = best_model_list[nfold]
                with torch.no_grad():
                    ftmodel.eval()
                    val_preds = []; val_labels = []
                    for i, batch in enumerate(val_list[nfold]):
                        outputs = ftmodel(batch['drug'])
                        val_preds.append(outputs.detach().cpu().numpy())
                        val_labels.append(batch['label'].detach().cpu().numpy())
                    val_preds_all.extend(list(np.vstack(val_preds).squeeze()))
                    val_labels_all.extend(list(np.vstack(val_labels).squeeze()))

                    test_preds = []; test_labels = []
                    for i, batch in enumerate(testft_loader):
                        outputs = ftmodel(batch['drug'])
                        test_preds.append(outputs.detach().cpu().numpy())
                        test_labels.append(batch['label'].detach().cpu().numpy())
                    test_preds_all.append(np.vstack(test_preds))

            val_r2 = r2_score(val_labels_all, val_preds_all)
            val_mse = mean_squared_error(val_labels_all, val_preds_all)
            print('val r2: %.4f, val mse: %.4f' % (val_r2, val_mse))
            test_preds_all = np.mean(np.array(test_preds_all), 0)
            test_r2 = r2_score(np.vstack(test_labels), test_preds_all)
            test_mse = mean_squared_error(np.vstack(test_labels), test_preds_all)
            print('test r2: %.4f, test mse: %.4f' % (test_r2, test_mse))

            preds_c.extend(list(test_preds_all.squeeze())); labels_c.extend(np.vstack(test_labels).squeeze())
            preds_c_val.extend(val_preds_all); labels_c_val.extend(val_labels_all)

        print(cname, 'val r2: %.4f, val mse: %.4f' % (r2_score(labels_c_val, preds_c_val), mean_squared_error(labels_c_val, preds_c_val)))
        print(cname, ' r2: %.4f, mse: %.4f' % (r2_score(labels_c, preds_c), mean_squared_error(labels_c, preds_c)))
        preds.extend(preds_c); labels.extend(labels_c)
        preds_val.extend(preds_c_val); labels_val.extend(labels_c_val)
    
    total_r2_val = r2_score(labels_val, preds_val)
    total_mse_val = mean_squared_error(labels_val, preds_val)
    print('total r2 val: %.4f, total mse val: %.4f' % (total_r2_val, total_mse_val))

    total_r2 = r2_score(labels, preds)
    total_mse = mean_squared_error(labels, preds)
    print('total r2: %.4f, total mse: %.4f' % (total_r2, total_mse))
