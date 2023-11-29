from torch.utils.data import DataLoader, Subset
import torch
from torch.nn import functional as F
from utils import * 
from metrics import * 
from argparse import ArgumentParser
import numpy as np
from dataloader import perturbDataset, DataCollator_Contrast_NoAug
from model import MoCoPair, SimCLRPair
import sys
import os
import warnings
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
    parser.add_argument('--logitdim', type=int, default=97)

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

    ## evaluation params
    parser.add_argument('--proj', type=str, default='none') # none, linear, mlp
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataparallel', type=str2bool, default=False)
    parser.add_argument('--model_dir', type=str) 
    parser.add_argument('--mocoenc', type=str, default='kk') # qq, kk, qk, kq

    args = parser.parse_args()

    stdoutOrigin=sys.stdout 
    if not os.path.exists('results/evalacc'):
        os.makedirs('results/evalacc')
    sys.stdout = open(f"results/evalacc/{args.model_dir}.txt", "w")

    print(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Use GPU: %s" % torch.cuda.is_available())
    print(args.model_dir)
    metrics_dict_whole = {'Top1Acc0': TopNAccuracy0(1), 'Top5Acc0': TopNAccuracy0(5), 'Top10Acc0': TopNAccuracy0(10)}
    
    inst_info_drug, drug_repr_dict = load_files()
    sel_keys = list(inst_info_drug.keys())
    inst_info_drug = select_inst(inst_info_drug, sel_keys)

    scaler_drug = StandardScaler(mean=torch.from_numpy(np.mean(np.array(list(drug_repr_dict.values())), 0)),
                                std=torch.from_numpy(np.std(np.array(list(drug_repr_dict.values())), 0)))
    drug_repr_dict = {k: torch.clip(scaler_drug.transform(v), -2.5, 2.5) for k, v in dict2torch(drug_repr_dict).items()} # 0.01 and 0.99 quantile
    
    whole_dataset = perturbDataset(inst_info_drug)
    print(len(whole_dataset.diff_ges))
    diff_ges_agg = np.vstack([item for sublist in whole_dataset.diff_ges for item in sublist])
    scaler_clip = 1
    diffges_clip = np.clip(diff_ges_agg, -scaler_clip, scaler_clip)
    scaler = StandardScaler(mean=torch.from_numpy(np.mean(diffges_clip, 0)).to(torch.float32),
                                std=torch.from_numpy(np.std(diffges_clip, 0)).to(torch.float32))

    data_collator_noaug = DataCollator_Contrast_NoAug(replicate=False)
    if args.model_dir != 'rdm':
        save_dir_sub = [f for f in os.listdir('/home/wangchy/InfoCORE/CP/model_save/') if f.endswith(args.model_dir)][0]
        save_dir = os.path.join('/home/wangchy/InfoCORE/CP/model_save/', save_dir_sub)
        train_labels = np.load(os.path.join(save_dir, 'train_labels.npy'))
    else:
        train_labels = np.load('/home/wangchy/InfoCORE/CP/data/training_CDRP_Bray2017/cp_train_labels.npy')
    print(train_labels[:10])
    train_idx = [i for i, item in enumerate(whole_dataset) if item['label'] in train_labels]
    valid_idx = list(set(range(len(whole_dataset))) - set(train_idx))
    train_dataset = Subset(whole_dataset, train_idx); print(len(train_dataset))
    valid_dataset = Subset(whole_dataset, valid_idx); print(len(valid_dataset))
    
    batches = [d['batch'] for d in valid_dataset]
    batchidx_dict = {}
    for i, b in enumerate(batches):
        if b not in batchidx_dict.keys():
            batchidx_dict[b] = [i]
        else:
            batchidx_dict[b].append(i)
    delete_idx = []
    for b in batchidx_dict.keys():
        if len(set(batchidx_dict[b])) < 20:
            delete_idx.extend(batchidx_dict[b])
    batchidx_dict = {b: list(set(batchidx_dict[b])) for b in batchidx_dict.keys() if len(set(batchidx_dict[b])) > 20}
    valid_bybatch_subset_dict = {b: Subset(valid_dataset, batchidx_dict[b]) for b in batchidx_dict.keys()}
    valid_batch_loader_dict = {k: DataLoader(v, batch_size=1000, shuffle=False, collate_fn=data_collator_noaug) for k, v in valid_bybatch_subset_dict.items()}
    
    valid_dataset = Subset(valid_dataset, list(set(range(len(valid_dataset))) - set(delete_idx))); print(len(valid_dataset))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator_noaug)
    
    if args.model == 'MoCo':
        model = MoCoPair(args, drug_repr_dict, device).to(device)
    elif args.model == 'SimCLR':
        model = SimCLRPair(args, drug_repr_dict, device).to(device)

    if args.model_dir != 'rdm':
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar'), map_location=device))
    if args.dataparallel:
        model = model.module
        
    def eval_whole(the_loader, this_model):
        this_model.eval()
        c_ges_latent = []; c_dc_latent = []; c_drugs = []
        with torch.no_grad():
            for batch in the_loader:
                drug = batch['drug']; diff_ges = batch['diff_ges']
                diff_ges = scaler.transform(torch.clip(diff_ges, -scaler_clip, scaler_clip)).to(device)
                if args.model =='MoCo':
                    q_ges_latent, q_dc_latent, _, _ = this_model.encoder_q(diff_ges, drug)
                    k_ges_latent, k_dc_latent, _, _ = this_model.encoder_k(diff_ges, drug)
                    if args.mocoenc == 'qq': ges_latent = q_ges_latent; dc_latent = q_dc_latent
                    elif args.mocoenc == 'kk': ges_latent = k_ges_latent; dc_latent = k_dc_latent
                    elif args.mocoenc == 'qk': ges_latent = q_ges_latent; dc_latent = k_dc_latent
                    elif args.mocoenc == 'kq': ges_latent = k_ges_latent; dc_latent = q_dc_latent
                elif args.model == 'SimCLR':
                    ges_latent, dc_latent, _, _ = this_model.encoder(diff_ges, drug)
                else:
                    raise NotImplementedError('the given model not implemented yet')
                c_ges_latent.append(ges_latent.cpu()); c_dc_latent.append(dc_latent.cpu()); c_drugs.extend(batch['drug'])
        c_ges_latent = torch.cat(c_ges_latent); c_dc_latent = torch.cat(c_dc_latent)
        if args.model_dir == 'rdm':
            metric = {k: 0 for k in metrics_dict_whole.keys()}
            metric['Top1Acc0'] = 1/c_ges_latent.shape[0]
            metric['Top5Acc0'] = 5/c_ges_latent.shape[0]
            metric['Top10Acc0'] = 10/c_ges_latent.shape[0]
        else:
            metric = {k: v(c_ges_latent, c_dc_latent).item() for k, v in metrics_dict_whole.items()}
        return metric
    
    valid_acc = eval_whole(valid_loader, model)

    def eval_bybatch(batch_loader_dict, this_model):
        metric_results = {k: 0 for k in metrics_dict_whole.keys()}
        n_total = 0
        for c, loader in batch_loader_dict.items():
            for i, batch in enumerate(loader):
                if len(batch['drug']) < 2:
                    continue
                drug = batch['drug']; diff_ges = batch['diff_ges']
                if args.model_dir == 'rdm':
                    metric_results['Top1Acc0'] += 1
                    metric_results['Top5Acc0'] += 5
                    metric_results['Top10Acc0'] += 10
                    n_total += len(batch['drug'])
                    continue
                diff_ges = scaler.transform(torch.clip(diff_ges, -scaler_clip, scaler_clip)).to(device)

                if args.model == 'MoCo':
                    q_ges_latent, q_dc_latent, _, _ = this_model.encoder_q(diff_ges, drug)
                    k_ges_latent, k_dc_latent, _, _ = this_model.encoder_k(diff_ges, drug)
                    if args.mocoenc == 'qq': ges_latent = q_ges_latent; dc_latent = q_dc_latent
                    elif args.mocoenc == 'kk': ges_latent = k_ges_latent; dc_latent = k_dc_latent
                    elif args.mocoenc == 'qk': ges_latent = q_ges_latent; dc_latent = k_dc_latent
                    elif args.mocoenc == 'kq': ges_latent = k_ges_latent; dc_latent = q_dc_latent
                elif args.model == 'SimCLR':
                    ges_latent, dc_latent, _, _ = this_model.encoder(diff_ges, drug)
                else:
                    raise NotImplementedError('the given model not implemented yet')
                ges_latent = F.normalize(ges_latent, dim=1); dc_latent = F.normalize(dc_latent, dim=1)

                for k, v in metrics_dict_whole.items():
                    metric_results[k] += v(ges_latent, dc_latent).item() * ges_latent.shape[0]
                n_total += ges_latent.shape[0]
        metric_results = {k: v/n_total for k, v in metric_results.items()}
        return metric_results
    
    valid_batch_acc = eval_bybatch(valid_batch_loader_dict, model)
    print('by plate:')
    print(f'top 1: {round(valid_batch_acc["Top1Acc0"], 4)*100}, top 5: {round(valid_batch_acc["Top5Acc0"], 4)*100}, top 10: {round(valid_batch_acc["Top10Acc0"], 4)*100}')
    print('whole:')
    print(f'top 1: {round(valid_acc["Top1Acc0"], 4)*100}, top 5: {round(valid_acc["Top5Acc0"], 4)*100}, top 10: {round(valid_acc["Top10Acc0"], 4)*100}')

    sys.stdout.close()
    sys.stdout=stdoutOrigin
