from torch.utils.data import DataLoader, Subset, Dataset
import torch
from torch.nn import functional as F
from torch import nn
from utils import *
from metrics import *
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from model import MoCoPair, SimCLRPair
import os
import warnings
import copy
import deepchem as dc
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
warnings.filterwarnings("ignore")

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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epoches', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_dir', type=str) 
    parser.add_argument('--druglistname', type=str) 
    parser.add_argument('--criterion', type=str, default='auc')     
    parser.add_argument('--n_iter', type=int, default=3) # number of random runs
    parser.add_argument('--mocoenc', type=str, default='q') # q, k

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

    # download data from OGB
    ogbdatapath = '/home/wangchy/InfoCORE/CP/data/ogb_data'
    if not os.path.exists(ogbdatapath):
        os.makedirs(ogbdatapath)
    dataset_name = args.druglistname
    dataset = PygGraphPropPredDataset(name=dataset_name, root=ogbdatapath)
    split_idx = dataset.get_idx_split()
    dname = dataset_name.split('-')[0]+'_'+dataset_name.split('-')[1]
    smiles = pd.read_csv(os.path.join(ogbdatapath, f'{dname}/mapping/mol.csv.gz'))
    loader = dc.data.CSVLoader(list(smiles.columns[:-2]), feature_field='smiles', featurizer=dc.feat.Mol2VecFingerprint())
    dataset = loader.create_dataset(os.path.join(ogbdatapath, f'{dname}/mapping/mol.csv.gz'))

    druglist = dataset.ids
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

    if args.model_dir != 'rdm':
        save_dir_sub = [f for f in os.listdir('/home/wangchy/InfoCORE/CP/model_save/') if f.endswith(args.model_dir)][0]
        save_dir = os.path.join('/home/wangchy/InfoCORE/CP/model_save/', save_dir_sub)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar'), map_location=device))
        if args.dataparallel:
            model = model.module
        if args.model == 'MoCo':
            model.encoder_k.requires_grad_(True)

    class finetune(nn.Module):
        def __init__(self, model):
            super(finetune, self).__init__()
            self.model = model
            self.n_warmup_epochs = 1
            self.fc = nn.Linear(args.enc_hiddim, dataset.y.shape[1])
        def forward(self, d, epoch):
            if args.model == 'MoCo':
                if args.mocoenc == 'q':
                    demb = self.model.encoder_q.get_dc_embedding(d)
                elif args.mocoenc == 'k':
                    demb = self.model.encoder_k.get_dc_embedding(d)
            elif args.model == 'SimCLR':
                demb = self.model.encoder.get_dc_embedding(d)
            else:
                demb = self.model.get_dc_embedding(d)
            if epoch < self.n_warmup_epochs:
                return self.fc(demb.detach())
            else:
                return self.fc(demb)

    class ftdataset(Dataset):
        def __init__(self, dataset):
            self.drugs = dataset.ids
            self.y = dataset.y
        def __len__(self):
            return len(self.drugs)
        def __getitem__(self, idx):
            return {'drug': self.drugs[idx], 'label': torch.from_numpy(self.y[idx]).to(torch.float32).to(device)}

    trainft_dataset = ftdataset(dataset.select(split_idx['train'].numpy()))
    valft_dataset = ftdataset(dataset.select(split_idx['valid'].numpy()))
    testft_dataset = ftdataset(dataset.select(split_idx['test'].numpy()))
    trainft_loader = DataLoader(trainft_dataset, batch_size=args.batch_size, shuffle=True)
    valft_loader = DataLoader(valft_dataset, batch_size=args.batch_size, shuffle=False)
    testft_loader = DataLoader(testft_dataset, batch_size=args.batch_size, shuffle=False)
    trainvalft_dataset = ftdataset(dataset.select(np.concatenate([split_idx['train'].numpy(), split_idx['valid'].numpy()])))

    evaluator = Evaluator(name = args.druglistname)
    criterion = nn.BCEWithLogitsLoss()

    def smi2scaffold(smi):
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=True)
        except:
            return smi
    
    results_list = []; results_list_val = []
    for iter in range(args.n_iter):
        splitter = GroupKFold(n_splits=5)
        splitted = splitter.split(trainvalft_dataset, trainvalft_dataset.y, [smi2scaffold(d) for d in trainvalft_dataset.drugs])
        for nfold, (train_idx, val_idx) in enumerate(splitted):
            ftmodel = finetune(copy.deepcopy(model)).to(device)
            optimizer_ft = torch.optim.Adam(ftmodel.parameters(), lr=args.lr)
            if args.druglistname == 'ogbg-molbbbp':
                trainft_dataset = Subset(trainvalft_dataset, train_idx)
                valft_dataset = Subset(trainvalft_dataset, val_idx)
            else:
                trainft_dataset = ftdataset(dataset.select(split_idx['train'].numpy()))
                valft_dataset = ftdataset(dataset.select(split_idx['valid'].numpy()))
            trainft_loader = DataLoader(trainft_dataset, batch_size=args.batch_size, shuffle=True)
            valft_loader = DataLoader(valft_dataset, batch_size=args.batch_size, shuffle=False)

            best_val_auc = 0.0
            best_loss = 1e10
            for epoch in range(args.n_epoches):
                running_loss = 0.0
                ftmodel.train()
                for i, batch in enumerate(trainft_loader):
                    optimizer_ft.zero_grad()
                    outputs = ftmodel(batch['drug'], epoch)
                    loss = criterion(outputs, batch['label'].to(device))
                    loss.backward()
                    optimizer_ft.step()
                    running_loss += loss.item()
                running_loss /= (i+1)
                ftmodel.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_preds = []; val_labels = []
                    for i, batch in enumerate(valft_loader):
                        outputs = ftmodel(batch['drug'], epoch)
                        loss = criterion(outputs, batch['label'].to(device))
                        val_loss += loss.item()
                        val_preds.append(F.sigmoid(outputs).detach().cpu().numpy())
                        val_labels.append(batch['label'].detach().cpu().numpy())
                    input_dict = {'y_true': np.vstack(val_labels), 'y_pred': np.vstack(val_preds)}
                    val_result_dict = evaluator.eval(input_dict)
                    val_loss /= (i+1)
                    if ((args.criterion=='auc' and val_result_dict['rocauc'] > best_val_auc) or (args.criterion=='loss' and val_loss < best_loss)):
                        best_val_auc = val_result_dict['rocauc']
                        best_loss = val_loss
                        best_epoch = epoch
                        os.makedirs(f'/home/wangchy/InfoCORE/CP/ftmodel_save/{args.druglistname}/{args.model_dir}_{args.lr}_{args.n_epoches}', exist_ok=True)
                        torch.save(ftmodel.state_dict(), os.path.join(f'/home/wangchy/InfoCORE/CP/ftmodel_save/{args.druglistname}/{args.model_dir}_{args.lr}_{args.n_epoches}', f'model_{nfold}.tar'))

                    test_loss = 0.0
                    test_preds = []; test_labels = []
                    for i, batch in enumerate(testft_loader):
                        outputs = ftmodel(batch['drug'], epoch)
                        loss = criterion(outputs, batch['label'].to(device))
                        test_loss += loss.item()
                        test_preds.append(F.sigmoid(outputs).detach().cpu().numpy())
                        test_labels.append(batch['label'].detach().cpu().numpy())
                    input_dict = {'y_true': np.vstack(test_labels), 'y_pred': np.vstack(test_preds)}
                    test_result_dict = evaluator.eval(input_dict)
                    test_loss /= (i+1)

        test_preds_all = []; val_preds_all = []
        for nfold in range(5):
            ftmodel.load_state_dict(torch.load(os.path.join(f'/home/wangchy/InfoCORE/CP/ftmodel_save/{args.druglistname}/{args.model_dir}_{args.lr}_{args.n_epoches}', f'model_{nfold}.tar')))
            ftmodel.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_preds = []; val_labels = []
                for i, batch in enumerate(valft_loader):
                    outputs = ftmodel(batch['drug'], 0)
                    loss = criterion(outputs, batch['label'].to(device))
                    val_loss += loss.item()
                    val_preds.append(F.sigmoid(outputs).detach().cpu().numpy())
                    val_labels.append(batch['label'].detach().cpu().numpy())
                input_dict = {'y_true': np.vstack(val_labels), 'y_pred': np.vstack(val_preds)}
                val_result_dict = evaluator.eval(input_dict)
                val_loss /= (i+1)
                val_preds_all.append(np.vstack(val_preds))

                test_loss = 0.0
                test_preds = []; test_labels = []
                for i, batch in enumerate(testft_loader):
                    outputs = ftmodel(batch['drug'], 0)
                    loss = criterion(outputs, batch['label'].to(device))
                    test_loss += loss.item()
                    test_preds.append(F.sigmoid(outputs).detach().cpu().numpy())
                    test_labels.append(batch['label'].detach().cpu().numpy())
                input_dict = {'y_true': np.vstack(test_labels), 'y_pred': np.vstack(test_preds)}
                test_result_dict = evaluator.eval(input_dict)
                test_loss /= (i+1)
                test_preds_all.append(np.vstack(test_preds))
        
        val_preds_all = np.mean(np.array(val_preds_all), 0)
        input_dict = {'y_true': np.vstack(val_labels), 'y_pred': val_preds_all}
        results_list_val.append(list(evaluator.eval(input_dict).values())[0])

        test_preds_all = np.mean(np.array(test_preds_all), 0)
        input_dict = {'y_true': np.vstack(test_labels), 'y_pred': test_preds_all}
        results_list.append(list(evaluator.eval(input_dict).values())[0])
         
    print(np.mean(results_list_val), np.std(results_list_val), np.mean(results_list), np.std(results_list))
