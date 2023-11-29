from torch.utils.data import DataLoader, Subset
import torch
from utils import *
from metrics import *
from argparse import ArgumentParser
import numpy as np
from dataloader import *
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
    parser.add_argument('--classify_intldim', type=int, default=2048) # 0 if linear classifier
    parser.add_argument('--clf_steps', type=int, default=2)
    parser.add_argument('--infocore', type=str2bool, default=False)
    parser.add_argument('--mi_weight_coef', type=float, default=1.0) # \hat{p}(x_b^1|z_t^1,z_g^i) = mi_weight_coef * \hat{p}(x_b^1|z_t^1) + \hat{p}(x_b^1|z_g^i)
    parser.add_argument('--anchor_logit', type=str, default='q') # q or k for MoCo
    parser.add_argument('--rewgrad_coef', type=float, default=0.0) # weight of adv gradient
    parser.add_argument('--logitdim', type=int, default=1019) # number of categorical batches
    
    ## MLP params
    # drug screen embedding (X_g, Z_g)
    parser.add_argument('--gene_dim', type=int, default=978)
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
    parser.add_argument('--cellproj', type=str2bool, default=True)
    parser.add_argument('--cellproj_epoch', type=int, default=5)
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
    metrics_dict_unseen = {'Top1Acc0': TopNAccuracy0(1), 'Top5Acc0': TopNAccuracy0(5), 'Top10Acc0': TopNAccuracy0(10)}
    
    inst_info_drug, ctl_dict_mean, id_smiles, drug_repr_dict = load_files()
    sel_keys = selkey_subsetboth_core(inst_info_drug)
    inst_info_drug = select_inst(inst_info_drug, sel_keys)

    scaler_drug = StandardScaler(mean=torch.from_numpy(np.mean(np.array(list(drug_repr_dict.values())), 0)),
                                std=torch.from_numpy(np.std(np.array(list(drug_repr_dict.values())), 0)))
    drug_repr_dict = {k: torch.clip(scaler_drug.transform(v), -2.5, 2.5) for k, v in dict2torch(drug_repr_dict).items()}

    whole_dataset = perturbDataset(inst_info_drug, ctl_dict_mean, id_smiles, drug_repr_dict)
    print(len(whole_dataset.diff_ges))
    diff_ges_agg = np.vstack([item for sublist in whole_dataset.diff_ges for item in sublist])
    scaler_clip = 1.5
    diffges_clip = np.clip(diff_ges_agg, -scaler_clip, scaler_clip)

    scaler = StandardScaler(mean=torch.from_numpy(np.mean(diffges_clip, 0)),
                            std=torch.from_numpy(np.std(diffges_clip, 0)))

    data_collator_noaug = DataCollator_Contrast_NoAug(replicate=False)
    if args.model_dir != 'rdm':
        save_dir_sub = [f for f in os.listdir('/home/wangchy/InfoCORE/GE/model_save/') if f.endswith(args.model_dir)][0]
        save_dir = os.path.join('/home/wangchy/InfoCORE/GE/model_save/', save_dir_sub)
        train_labels = np.load(os.path.join(save_dir, 'train_labels.npy'))
    else:
        train_labels = np.load('/home/wangchy/InfoCORE/GE/data/training_LINCS/subsetbothtrain_labels_bydrug.npy')
    print(train_labels[:10])
    train_idx = [i for i, item in enumerate(whole_dataset) if item['label'] in train_labels]
    valid_idx = list(set(range(len(whole_dataset))) - set(train_idx))
    train_dataset = Subset(whole_dataset, train_idx); print(len(train_dataset))
    valid_dataset = Subset(whole_dataset, valid_idx); print(len(valid_dataset))
    
    batches = [d['batch'] for d in valid_dataset]
    batchidx_dict = {}
    for i, batchid in enumerate(batches):
        for b in batchid:
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
    valid_cell_dataset = dataset2cellsubset(valid_dataset)
    valid_cell_loader_dict = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator_noaug) for k, v in valid_cell_dataset.items()}
    valid_drugs = list(set([item['drug'] for item in valid_dataset])); print(len(valid_drugs))
    cells = list(set(whole_dataset.celltypes))

    if args.model == 'MoCo':
        model = MoCoPair(args, drug_repr_dict, device, cells=cells).to(device)
    elif args.model == 'SimCLR':
        model = SimCLRPair(args, drug_repr_dict, device).to(device)
    else:
        raise NotImplementedError('the given model not implemented yet')
    if args.dataparallel:
        model = data_parallel(model)

    if args.model_dir != 'rdm':
        model.load_state_dict(torch.load(os.path.join(save_dir, 'model.tar'), map_location=device))
    
    if args.dataparallel:
        model = model.module

    def eval_bycell_whole(cell_loader_dict, this_model, all_drugs=valid_drugs):
        cell_acc_unseen = {c: {k: 0 for k in ['num']+list(metrics_dict_unseen.keys())} for c in cells+['all']}
        this_model.eval()
        for c, loader in cell_loader_dict.items():
            c_ges_latent = []; c_dc_latent = []; c_drugs = []
            with torch.no_grad():
                for batch in loader:
                    cell = batch['celltype']; drug = batch['drug']; diff_ges = batch['diff_ges']
                    diff_ges = scaler.transform(torch.clip(diff_ges, -scaler_clip, scaler_clip)).to(device)
                    if args.model =='MoCo':
                        q_ges_latent, q_dc_latent, _, _ = this_model.encoder_q(diff_ges, cell, drug, epoch=100)
                        k_ges_latent, k_dc_latent, _, _ = this_model.encoder_k(diff_ges, cell, drug, epoch=100)
                        if args.mocoenc == 'qq': ges_latent = q_ges_latent; dc_latent = q_dc_latent
                        elif args.mocoenc == 'kk': ges_latent = k_ges_latent; dc_latent = k_dc_latent
                        elif args.mocoenc == 'qk': ges_latent = q_ges_latent; dc_latent = k_dc_latent
                        elif args.mocoenc == 'kq': ges_latent = k_ges_latent; dc_latent = q_dc_latent
                    elif args.model == 'SimCLR':
                        ges_latent, dc_latent, _, _ = this_model.encoder(diff_ges, cell, drug, epoch=100)
                    c_ges_latent.append(ges_latent.cpu()); c_dc_latent.append(dc_latent.cpu()); c_drugs.extend(batch['drug'])
            c_ges_latent = torch.cat(c_ges_latent); c_dc_latent = torch.cat(c_dc_latent)

            newdrugs = list(set(all_drugs) - set(c_drugs))
            if len(newdrugs) > 0:
                dcls = []
                with torch.no_grad():
                    for ii in range(0, len(newdrugs), args.batch_size):
                        end = np.min([ii+args.batch_size, len(newdrugs)])
                        if args.model == 'MoCo':
                            if args.mocoenc == 'qq' or args.mocoenc == 'kq':
                                dcls.append(this_model.encoder_q.dcemb2latent(this_model.encoder_q.get_dc_embedding(newdrugs[ii:end]), [c]*(end-ii), epoch=100).cpu())
                            elif args.mocoenc == 'kk' or args.mocoenc == 'qk':
                                dcls.append(this_model.encoder_k.dcemb2latent(this_model.encoder_k.get_dc_embedding(newdrugs[ii:end]), [c]*(end-ii), epoch=100).cpu())
                        elif args.model == 'SimCLR':
                            dcls.append(this_model.encoder.dcemb2latent(this_model.encoder.get_dc_embedding(newdrugs[ii:end]), [c]*(end-ii), epoch=100).cpu())
                dc_latent_new = torch.cat(dcls)
                c_dc_unseen = torch.cat([c_dc_latent, dc_latent_new])
            else:
                c_dc_unseen = c_dc_latent
            cell_acc_unseen[c]['num'] = c_dc_unseen.shape[0]
            if args.model_dir == 'rdm':
                cell_acc_unseen[c]['Top1Acc0'] = 1/cell_acc_unseen[c]['num']
                cell_acc_unseen[c]['Top5Acc0'] = 5/cell_acc_unseen[c]['num']
                cell_acc_unseen[c]['Top10Acc0'] = 10/cell_acc_unseen[c]['num']
            else:
                for k, v in metrics_dict_unseen.items():
                    cell_acc_unseen[c][k] = v(c_ges_latent, c_dc_unseen).item()
            for k in metrics_dict_unseen.keys():
                cell_acc_unseen['all'][k] = np.sum([cell_acc_unseen[tcell]['num']*cell_acc_unseen[tcell][k] for tcell in cells])/np.sum([cell_acc_unseen[tcell]['num'] for tcell in cells])
        return cell_acc_unseen
    
    valid_cell_acc_unseen = eval_bycell_whole(valid_cell_loader_dict, model, all_drugs=valid_drugs)

    def eval_bybatch(batch_loader_dict, this_model):
        metric_results = {k: 0 for k in metrics_dict_unseen.keys()}
        n_total = 0
        this_model.eval()
        for c, loader in batch_loader_dict.items():
            for i, batch in enumerate(loader):
                if len(batch['celltype']) < 2:
                    continue
                cell = batch['celltype']; drug = batch['drug']; diff_ges = batch['diff_ges']
                if args.model_dir == 'rdm':
                    metric_results['Top1Acc0'] += 1
                    metric_results['Top5Acc0'] += 5
                    metric_results['Top10Acc0'] += 10
                    n_total += len(batch['celltype'])
                    continue
                diff_ges = scaler.transform(torch.clip(diff_ges, -scaler_clip, scaler_clip)).to(device)
                if args.model == 'MoCo':
                    q_ges_latent, q_dc_latent, _, _ = this_model.encoder_q(diff_ges, cell, drug, epoch=100)
                    k_ges_latent, k_dc_latent, _, _ = this_model.encoder_k(diff_ges, cell, drug, epoch=100)
                    if args.mocoenc == 'qq': ges_latent = q_ges_latent; dc_latent = q_dc_latent
                    elif args.mocoenc == 'kk': ges_latent = k_ges_latent; dc_latent = k_dc_latent
                    elif args.mocoenc == 'qk': ges_latent = q_ges_latent; dc_latent = k_dc_latent
                    elif args.mocoenc == 'kq': ges_latent = k_ges_latent; dc_latent = q_dc_latent
                elif args.model == 'SimCLR':
                    ges_latent, dc_latent, _, _ = this_model.encoder(diff_ges, cell, drug, epoch=100)
                for k, v in metrics_dict_unseen.items():
                    metric_results[k] += v(ges_latent, dc_latent).item() * ges_latent.shape[0]
                n_total += ges_latent.shape[0]
        metric_results = {k: v/n_total for k, v in metric_results.items()}
        return metric_results
    
    valid_batch_acc = eval_bybatch(valid_batch_loader_dict, model)
    print('batch:')
    print(f'top 1: {round(valid_batch_acc["Top1Acc0"], 4)*100}, top 5: {round(valid_batch_acc["Top5Acc0"], 4)*100}, top 10: {round(valid_batch_acc["Top10Acc0"], 4)*100}')
    print('whole:')
    print(f'top 1: {round(valid_cell_acc_unseen["all"]["Top1Acc0"], 4)*100}, top 5: {round(valid_cell_acc_unseen["all"]["Top5Acc0"], 4)*100}, top 10: {round(valid_cell_acc_unseen["all"]["Top10Acc0"], 4)*100}')
    sys.stdout.close()
    sys.stdout=stdoutOrigin
