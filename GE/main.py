from torch.utils.data import DataLoader, Subset
import torch
from torch.nn import functional as F
from utils import *
from metrics import *
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from dataloader import *
from model import MoCoPair, SimCLRPair
import random
import wandb
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

    ## Data Augmemtation params
    parser.add_argument('--replicate', type=str2bool, default=True)
    parser.add_argument('--dirmixup', type=str2bool, default=True)
    parser.add_argument('--noise', type=str2bool, default=True)
    parser.add_argument('--dir_hyp', type=float, default=0.5)
    parser.add_argument('--noise_scale', type=float, default=0.6)
    parser.add_argument('--replicate_valid', type=str2bool, default=False) # use replicate or mean for validation

    ## training and evaluation params
    parser.add_argument('--proj', type=str, default='none') # none, linear, mlp
    parser.add_argument('--train_bybatch', type=str2bool, default=False) # True for CCL
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_clf', type=float, default=1e-4) 
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--steps', type=float, default=[30, 60, 90, 120, 150, 200, 250, 300])
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--n_epoches', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--wandb', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_bycell', type=str, default='rdm+bycell') # rdmonly, rdm+bycell
    parser.add_argument('--cellproj', type=str2bool, default=True)
    parser.add_argument('--cellproj_epoch', type=int, default=5)
    parser.add_argument('--dataparallel', type=str2bool, default=False)

    args = parser.parse_args()

    if args.wandb:
        run = wandb.init(
            project="DrugRepMain3",
            name='new run',
            notes='new run',
            save_code=True,
            config=args,
        )
    print(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Use GPU: %s" % torch.cuda.is_available())
    
    metrics_dict = {'DimCov': DimensionCovariance(), 'BatchVar': BatchVariance(), 'Alignment': Alignment(), 'Uniformity': Uniformity(), 
            'PosSim': PositiveSimilarity(), 'NegSim': NegativeSimilarity(), 'Top1Acc0': TopNAccuracy0(1),'Top5Acc0': TopNAccuracy0(5)}
    metrics_dict_names = list(metrics_dict.keys())
    if args.infocore:
        metrics_dict_names += ['H(w)_ges', 'H(w)_dc'] # entropy of weights
    
    inst_info_drug, ctl_dict_mean, id_smiles, drug_repr_dict = load_files()
    sel_keys = selkey_subsetboth_core(inst_info_drug)
    inst_info_drug = select_inst(inst_info_drug, sel_keys)
    
    if args.infocore:
        with open("/home/wangchy/InfoCORE/GE/data/training_LINCS/labellogits_dict.pkl", "rb") as fp:
            labellogits_dict = pickle.load(fp) # batch label (1/n_replicate for each replicate batch)

    scaler_drug = StandardScaler(mean=torch.from_numpy(np.mean(np.array(list(drug_repr_dict.values())), 0)),
                                std=torch.from_numpy(np.std(np.array(list(drug_repr_dict.values())), 0)))
    drug_repr_dict = {k: torch.clip(scaler_drug.transform(v), -2.5, 2.5) for k, v in dict2torch(drug_repr_dict).items()} # 0.01 and 0.99 quantile

    whole_dataset = perturbDataset(inst_info_drug, ctl_dict_mean, id_smiles, drug_repr_dict)
    print(f'{len(whole_dataset.diff_ges)} (drug-cell line) pairs')
    diff_ges_agg = np.vstack([item for sublist in whole_dataset.diff_ges for item in sublist])
    scaler_clip = 1.5
    diffges_clip = np.clip(diff_ges_agg, -scaler_clip, scaler_clip)
    scaler = StandardScaler(mean=torch.from_numpy(np.mean(diffges_clip, 0)),
                            std=torch.from_numpy(np.std(diffges_clip, 0)))

    data_collator_aug = DataCollator_Contrast(args.replicate, args.dirmixup, args.noise, args.dir_hyp, args.noise_scale)
    data_collator_noaug = DataCollator_Contrast_NoAug(args.replicate_valid)
    train_idx_d, val_idx_d, test_idx_d = new_drug_sampler(whole_dataset.drugs, seed=0) # train-val-test split by drug
    train_dataset = Subset(whole_dataset, train_idx_d); val_dataset = Subset(whole_dataset, val_idx_d); test_dataset = Subset(whole_dataset, test_idx_d)
    train_labels = [item['label'] for item in train_dataset]
    print(train_labels[:10])
    val_labels = [item['label'] for item in val_dataset]
    test_labels = [item['label'] for item in test_dataset]
    print(val_labels[:10])
    
    if args.wandb:
        s_dir = '/home/wangchy/InfoCORE/GE/model_save/' + wandb.run.dir.split('/')[-2]
        if not os.path.exists(s_dir): os.mkdir(s_dir)
        np.save(os.path.join(s_dir, 'train_labels.npy'), train_labels)
    
    torch.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_aug, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_noaug, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_noaug, drop_last=True)

    train_cell_dataset = dataset2cellsubset(train_dataset); val_cell_dataset = dataset2cellsubset(val_dataset); test_cell_dataset = dataset2cellsubset(test_dataset)
    train_cell_loader_dict = {k: DataLoader(v, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_aug) for k, v in train_cell_dataset.items()}
    val_cell_loader_dict = {k: DataLoader(v, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_noaug) for k, v in val_cell_dataset.items()}
    test_cell_loader_dict = {k: DataLoader(v, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_noaug) for k, v in test_cell_dataset.items()}
    
    train_batch_dataset = dataset2batchsubset(train_dataset); val_batch_dataset = dataset2batchsubset(val_dataset); test_batch_dataset = dataset2batchsubset(test_dataset)
    train_batch_loader_dict = {k: DataLoader(v, batch_size=512, shuffle=True, collate_fn=data_collator_aug) for k, v in train_batch_dataset.items()}
    val_batch_loader_dict = {k: DataLoader(v, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_noaug) for k, v in val_batch_dataset.items()}
    test_batch_loader_dict = {k: DataLoader(v, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator_noaug) for k, v in test_batch_dataset.items()}
    
    cells = list(set(whole_dataset.celltypes)) # ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP', 'HEPG2']

    def trainloader_bycell(cell_loader_dict):
        mixed_bycell_loader = [batch for loader in cell_loader_dict.values() for batch in loader]
        mixed_bycell_loader = [b for b in mixed_bycell_loader if len(b['celltype']) >= 10]
        random.shuffle(mixed_bycell_loader)
        return mixed_bycell_loader
    if args.train_bycell == 'rdm+bycell': train_loader_bycell = trainloader_bycell(train_cell_loader_dict)
    
    def trainloader_bybatch(batch_loader_dict):
        mixed_bybatch_loader = [batch for loader in batch_loader_dict.values() for batch in loader]
        mixed_bybatch_loader = [b for b in mixed_bybatch_loader if len(b['celltype']) >= 10]
        random.shuffle(mixed_bybatch_loader)
        return mixed_bybatch_loader
    if args.train_bybatch: train_loader_bybatch = trainloader_bybatch(train_batch_loader_dict)

    if args.model == 'MoCo':
        model = MoCoPair(args, drug_repr_dict, device, cells=cells).to(device)
    elif args.model == 'SimCLR':
        model = SimCLRPair(args, drug_repr_dict, device).to(device)
    else:
        raise NotImplementedError('model not implemented')
    if args.dataparallel:
        model = data_parallel(model)

    if args.infocore:
        if args.model == 'MoCo':
            if args.dataparallel:
                optimizer = torch.optim.Adam(model.module.encoder_q.parameters(), lr=args.lr, eps=args.epsilon, betas=(0.5, 0.999))
                optimizer_mi_clf = torch.optim.Adam(model.module.clf_q.parameters(), lr=args.lr_clf, eps=args.epsilon, betas=(0.5, 0.999))
            else:
                optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=args.lr, eps=args.epsilon, betas=(0.5, 0.999))
                optimizer_mi_clf = torch.optim.Adam(model.clf_q.parameters(), lr=args.lr_clf, eps=args.epsilon, betas=(0.5, 0.999))
        elif args.model == 'SimCLR':
            if args.dataparallel:
                optimizer = torch.optim.Adam(model.module.encoder.parameters(), lr=args.lr, eps=args.epsilon, betas=(0.5, 0.999))
                optimizer_mi_clf = torch.optim.Adam(model.module.clf.parameters(), lr=args.lr_clf, eps=args.epsilon, betas=(0.5, 0.999))
            else:
                optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, eps=args.epsilon, betas=(0.5, 0.999))
                optimizer_mi_clf = torch.optim.Adam(model.clf.parameters(), lr=args.lr_clf, eps=args.epsilon, betas=(0.5, 0.999))
        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
        lr_scheduler_mi_clf = torch.optim.lr_scheduler.MultiStepLR(optimizer_mi_clf, milestones=args.steps, gamma=args.lr_decay_ratio)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon, betas=(0.5, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)

    def model_getloss(this_model, diff_ges, cell, drug, index_tensor, epoch, geslogit, mode='eval', bycell=None):
        if args.model != 'MoCo':
            raise ValueError('model_getloss only works for MoCo')
        if mode == 'train':
            with torch.no_grad():
                if args.dataparallel:
                    this_model.module._momentum_update_key_encoder(epoch)
                    if args.cellproj and epoch >= args.cellproj_epoch and this_model.module.encoder_q.changeflag == False:
                        this_model.module.encoder_q.cellproj_copyweight(); this_model.module.encoder_k.cellproj_copyweight()
                        this_model.module.encoder_q.changeflag = True; this_model.module.encoder_k.changeflag = True
                else:
                    this_model._momentum_update_key_encoder(epoch)
                    if args.cellproj and epoch >= args.cellproj_epoch and this_model.encoder_q.changeflag == False:
                        this_model.encoder_q.cellproj_copyweight(); this_model.encoder_k.cellproj_copyweight()
                        this_model.encoder_q.changeflag = True; this_model.encoder_k.changeflag = True
        if args.infocore:
            loss_ges, loss_dc, q_ges_latent, q_dc_latent, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, weights_ges, weights_dc, k_ges_latent, k_dc_latent, k_ges_logit, k_dc_logit = this_model(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, bycell=bycell)
            if args.dataparallel: 
                if mode == 'train': this_model.module.queue(k_ges_latent, k_dc_latent, k_ges_logit=k_ges_logit, k_dc_logit=k_dc_logit, bycell=bycell)
                return this_model.module.get_loss(loss_ges, loss_dc, q_ges_latent, q_dc_latent, clf_loss_ges=clf_loss_ges, clf_loss_dc=clf_loss_dc, clf_loss_ges_k=clf_loss_ges_k, clf_loss_dc_k=clf_loss_dc_k, weights_ges=weights_ges, weights_dc=weights_dc)
            else: 
                if mode == 'train': this_model.queue(k_ges_latent, k_dc_latent, k_ges_logit=k_ges_logit, k_dc_logit=k_dc_logit, bycell=bycell)
                return this_model.get_loss(loss_ges, loss_dc, q_ges_latent, q_dc_latent, clf_loss_ges=clf_loss_ges, clf_loss_dc=clf_loss_dc, clf_loss_ges_k=clf_loss_ges_k, clf_loss_dc_k=clf_loss_dc_k, weights_ges=weights_ges, weights_dc=weights_dc)
        else:
            loss_ges, loss_dc, q_ges_latent, q_dc_latent, k_ges_latent, k_dc_latent = this_model(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, bycell=bycell)
            if args.dataparallel: 
                if mode == 'train': this_model.module.queue(k_ges_latent, k_dc_latent, bycell=bycell, geslogit=geslogit)
                return this_model.module.get_loss(loss_ges, loss_dc, q_ges_latent, q_dc_latent)
            else: 
                if mode == 'train': this_model.queue(k_ges_latent, k_dc_latent, bycell=bycell, geslogit=geslogit)
                return this_model.get_loss(loss_ges, loss_dc, q_ges_latent, q_dc_latent)
            
    def model_getloss_simclr(this_model, diff_ges, cell, drug, index_tensor, epoch, geslogit, mode='eval'):
        if args.model != 'SimCLR':
            raise ValueError('model_getloss_simclr only works for SimCLR')
        if mode == 'train':
            with torch.no_grad():
                if args.dataparallel:
                    if args.cellproj and epoch >= args.cellproj_epoch and this_model.module.encoder.changeflag == False:
                        this_model.module.encoder.cellproj_copyweight()
                        this_model.module.encoder.changeflag = True
                else:
                    if args.cellproj and epoch >= args.cellproj_epoch and this_model.encoder.changeflag == False:
                        this_model.encoder.cellproj_copyweight()
                        this_model.encoder.changeflag = True
        if args.infocore:
            loss_ges, loss_dc, ges_latent, dc_latent, clf_loss_ges, clf_loss_dc, weights_ges, weights_dc, g_logit, d_logit = this_model(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit)
            if args.dataparallel:
                return this_model.module.get_loss(loss_ges, loss_dc, ges_latent, dc_latent, clf_loss_ges=clf_loss_ges, clf_loss_dc=clf_loss_dc, weights_ges=weights_ges, weights_dc=weights_dc)
            else:
                return this_model.get_loss(loss_ges, loss_dc, ges_latent, dc_latent, clf_loss_ges=clf_loss_ges, clf_loss_dc=clf_loss_dc, weights_ges=weights_ges, weights_dc=weights_dc)
        else:
            loss_ges, loss_dc, ges_latent, dc_latent = this_model(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit)
            if args.dataparallel:
                return this_model.module.get_loss(loss_ges, loss_dc, ges_latent, dc_latent)
            else:
                return this_model.get_loss(loss_ges, loss_dc, ges_latent, dc_latent)

    def train_batch(batch, model, epoch=None, bycell=None):
        cell = batch['celltype']; drug = batch['drug']; diff_ges = batch['diff_ges']; label = batch['label']
        diff_ges = scaler.transform(torch.clip(diff_ges, -scaler_clip, scaler_clip)).to(device)
        if args.infocore:
            geslogit = torch.Tensor(np.vstack([labellogits_dict[lb] for lb in label]).astype(np.float32)).to(device)
        else:
            geslogit = None
        if args.model == 'MoCo':
            index_tensor = torch.arange(len(cell)).to(diff_ges.device)
            if args.infocore:
                loss, ges_latent, dc_latent, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, weightentropy = model_getloss(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='train', bycell=bycell)
            else:
                loss, ges_latent, dc_latent = model_getloss(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='train', bycell=bycell) 
            metric = {k: v(ges_latent, dc_latent).item() for k, v in metrics_dict.items()}
            if args.infocore:
                metric['H(w)_ges'] = weightentropy[0]; metric['H(w)_dc'] = weightentropy[1]
                return loss, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, metric, ges_latent, dc_latent, geslogit
            else:
                return loss, metric
        elif args.model == 'SimCLR':
            index_tensor = torch.arange(len(cell)).to(diff_ges.device)
            if args.infocore:
                loss, ges_latent, dc_latent, clf_loss_ges, clf_loss_dc, weightentropy = model_getloss_simclr(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='train')
            else:
                loss, ges_latent, dc_latent = model_getloss_simclr(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='train')
            metric = {k: v(ges_latent, dc_latent).item() for k, v in metrics_dict.items()}
            if args.infocore:
                metric['H(w)_ges'] = weightentropy[0]; metric['H(w)_dc'] = weightentropy[1]
                return loss, clf_loss_ges, clf_loss_dc, metric, ges_latent, dc_latent, geslogit
            else:
                return loss, metric

    def eval_batch(batch, model, epoch=None, bycell=None):
        cell = batch['celltype']; drug = batch['drug']; diff_ges = batch['diff_ges']; label = batch['label']
        diff_ges = scaler.transform(torch.clip(diff_ges, -scaler_clip, scaler_clip)).to(device)
        if args.infocore:
            geslogit = torch.Tensor(np.vstack([labellogits_dict[lb] for lb in label]).astype(np.float32)).to(device)
        else:
            geslogit = None
        if args.model == 'MoCo':
            index_tensor = torch.arange(len(cell)).to(diff_ges.device)
            with torch.no_grad():
                if args.infocore:
                    loss, ges_latent, dc_latent, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, weightentropy = model_getloss(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='eval', bycell=bycell)
                else:
                    loss, ges_latent, dc_latent = model_getloss(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='eval', bycell=bycell)
            metric = {k: v(ges_latent, dc_latent).item() for k, v in metrics_dict.items()}
            if args.infocore:
                metric['H(w)_ges'] = weightentropy[0]; metric['H(w)_dc'] = weightentropy[1]
                return loss, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, metric
            else:
                return loss, metric
        elif args.model == 'SimCLR':
            index_tensor = torch.arange(len(cell)).to(diff_ges.device)
            if args.infocore:
                loss, ges_latent, dc_latent, clf_loss_ges, clf_loss_dc, weightentropy = model_getloss_simclr(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='eval')
            else:
                loss, ges_latent, dc_latent = model_getloss_simclr(model, diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, mode='eval')
            metric = {k: v(ges_latent, dc_latent).item() for k, v in metrics_dict.items()}
            if args.infocore:
                metric['H(w)_ges'] = weightentropy[0]; metric['H(w)_dc'] = weightentropy[1]
                return loss, clf_loss_ges, clf_loss_dc, metric
            else:
                return loss, metric

    def eval_bygroup(group_loader_dict, this_model, epoch=None):
        total_loss = 0
        metric_results = {k: 0 for k in metrics_dict_names}
        if args.infocore: 
            total_clf_loss_ges = 0; total_clf_loss_dc = 0
            total_clf_loss_ges_k = 0; total_clf_loss_dc_k = 0

        i_total = 0
        this_model.eval()
        for c, loader in group_loader_dict.items():
            for i, batch in enumerate(loader):
                the_cell = batch['celltype'][0]
                if len(batch['celltype']) < 10:
                    continue
                i_total += 1
                if args.model == 'MoCo':
                    if args.infocore:
                        loss, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, metric = eval_batch(batch, this_model, epoch=epoch, bycell=the_cell)
                        total_clf_loss_ges += clf_loss_ges.item(); total_clf_loss_dc += clf_loss_dc.item()
                        total_clf_loss_ges_k += clf_loss_ges_k.item(); total_clf_loss_dc_k += clf_loss_dc_k.item()
                    else:
                        loss, metric = eval_batch(batch, this_model, epoch=epoch, bycell=the_cell) 
                elif args.model == 'SimCLR':
                    if args.infocore:
                        loss, clf_loss_ges, clf_loss_dc, metric = eval_batch(batch, this_model, epoch=epoch, bycell=the_cell)
                        total_clf_loss_ges += clf_loss_ges.item(); total_clf_loss_dc += clf_loss_dc.item()
                    else:
                        loss, metric = eval_batch(batch, this_model, epoch=epoch, bycell=the_cell)
                total_loss += loss.item()
                for k, v in metric.items():
                    metric_results[k] += v
        metric_results = {k: v/(i_total+1) for k, v in metric_results.items()}
        total_loss = total_loss / (i_total+1)
        if args.infocore: 
            total_clf_loss_ges = total_clf_loss_ges / (i_total+1); total_clf_loss_dc = total_clf_loss_dc / (i_total+1)
            metric_results['clfloss_ges'] = total_clf_loss_ges; metric_results['clfloss_dc'] = total_clf_loss_dc
            total_clf_loss_ges_k = total_clf_loss_ges_k / (i_total+1); total_clf_loss_dc_k = total_clf_loss_dc_k / (i_total+1) 
            metric_results['clfloss_ges_k'] = total_clf_loss_ges_k; metric_results['clfloss_dc_k'] = total_clf_loss_dc_k
        return total_loss, metric_results

    best_val_acc = 0
    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        if args.train_bycell == 'rdmonly' or args.train_bycell == 'rdm+bycell': 
            epoch_loss_train = 0; metric_results = {k: 0 for k in metrics_dict_names}
        if args.train_bycell == 'rdm+bycell': 
            train_loader_bycell = trainloader_bycell(train_cell_loader_dict)
            epoch_loss_trainbc = 0; metric_resultsbc = {k: 0 for k in metrics_dict_names}
        if args.train_bybatch: epoch_loss_trainbb = 0; metric_resultsbb = {k: 0 for k in metrics_dict_names}
        if args.infocore: 
            epoch_clfgesloss_train = 0; epoch_clfdcloss_train = 0; epoch_clfgesloss_trainbc = 0; epoch_clfdcloss_trainbc = 0
            epoch_clfgesloss_train_k = 0; epoch_clfdcloss_train_k = 0; epoch_clfgesloss_trainbc_k = 0; epoch_clfdcloss_trainbc_k = 0

        if args.train_bybatch:
            train_loader_bybatch = trainloader_bybatch(train_batch_loader_dict)
            for i, batch in enumerate(train_loader_bybatch):
                optimizer.zero_grad()
                if args.model == 'MoCo':
                    raise ValueError('train_bybatch only works for SimCLR')
                elif args.model == 'SimCLR':
                    loss, metric = train_batch(batch, model, epoch=epoch)
                    total_loss = loss
                if epoch > 0:
                    (0.3 * total_loss).backward()
                    optimizer.step()
                epoch_loss_trainbb += loss.item()
                for k, v in metric.items():
                    metric_resultsbb[k] += v
            metric_resultsbb = {k: v/(i+1) for k, v in metric_resultsbb.items()}; epoch_loss_trainbb = epoch_loss_trainbb / (i+1)
            log_dict = {'trainbb_'+k: v for k, v in metric_resultsbb.items()}
            log_dict['trainbb_loss'] = epoch_loss_trainbb
            wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)
        
        else:
            if args.train_bycell == 'rdmonly':
                for i, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    if args.model == 'MoCo':
                        if args.infocore:
                            loss, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, metric, ges_latent, dc_latent, geslogit = train_batch(batch, model, epoch=epoch, bycell=None)
                            epoch_clfgesloss_train += clf_loss_ges.item(); epoch_clfdcloss_train += clf_loss_dc.item()
                            epoch_clfgesloss_train_k += clf_loss_ges_k.item(); epoch_clfdcloss_train_k += clf_loss_dc_k.item()
                        else:
                            loss, metric = train_batch(batch, model, epoch=epoch, bycell=None)
                    elif args.model == 'SimCLR':
                        if args.infocore:
                            loss, clf_loss_ges, clf_loss_dc, metric, ges_latent, dc_latent, geslogit = train_batch(batch, model, epoch=epoch, bycell=None)
                            epoch_clfgesloss_train += clf_loss_ges.item(); epoch_clfdcloss_train += clf_loss_dc.item()
                        else:
                            loss, metric = train_batch(batch, model, epoch=epoch, bycell=None)
                    total_loss = loss
                    if epoch > 0:
                        total_loss.backward()                     
                        optimizer.step()

                    if args.infocore:
                        for _ in range(args.clf_steps):
                            optimizer_mi_clf.zero_grad()
                            if args.model == 'MoCo':
                                if args.dataparallel: clf_loss_ges = F.cross_entropy(model.module.clf_q.get_g_logit(ges_latent.detach()), geslogit)
                                else: clf_loss_ges = F.cross_entropy(model.clf_q.get_g_logit(ges_latent.detach()), geslogit)
                            elif args.model == 'SimCLR':
                                if args.dataparallel: clf_loss_ges = F.cross_entropy(model.module.clf.get_g_logit(ges_latent.detach()), geslogit)
                                else: clf_loss_ges = F.cross_entropy(model.clf.get_g_logit(ges_latent.detach()), geslogit)
                            clf_loss_ges.backward()
                            optimizer_mi_clf.step()
                        for _ in range(args.clf_steps):
                            optimizer_mi_clf.zero_grad()
                            if args.model == 'MoCo':
                                if args.dataparallel: clf_loss_dc = F.cross_entropy(model.module.clf_q.get_d_logit(dc_latent.detach()), geslogit)
                                else: clf_loss_dc = F.cross_entropy(model.clf_q.get_d_logit(dc_latent.detach()), geslogit)
                            elif args.model == 'SimCLR':
                                if args.dataparallel: clf_loss_dc = F.cross_entropy(model.module.clf.get_d_logit(dc_latent.detach()), geslogit)
                                else: clf_loss_dc = F.cross_entropy(model.clf.get_d_logit(dc_latent.detach()), geslogit)
                            clf_loss_dc.backward()
                            optimizer_mi_clf.step()
                        
                    epoch_loss_train += loss.item()
                    for k, v in metric.items():
                        metric_results[k] += v

            elif args.train_bycell == 'rdm+bycell':
                for i, (batch, batchbc) in enumerate(zip(train_loader, train_loader_bycell)):
                    optimizer.zero_grad()
                    if args.model == 'MoCo':
                        if args.infocore:
                            loss, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, metric, ges_latent, dc_latent, geslogit = train_batch(batch, model, epoch=epoch, bycell=None)
                            epoch_clfgesloss_train += clf_loss_ges.item(); epoch_clfdcloss_train += clf_loss_dc.item()
                            epoch_clfgesloss_train_k += clf_loss_ges_k.item(); epoch_clfdcloss_train_k += clf_loss_dc_k.item()
                        else:
                            loss, metric = train_batch(batch, model, epoch=epoch, bycell=None)
                    elif args.model == 'SimCLR':
                        if args.infocore:
                            loss, clf_loss_ges, clf_loss_dc, metric, ges_latent, dc_latent, geslogit = train_batch(batch, model, epoch=epoch, bycell=None)
                            epoch_clfgesloss_train += clf_loss_ges.item(); epoch_clfdcloss_train += clf_loss_dc.item()
                        else:
                            loss, metric = train_batch(batch, model, epoch=epoch, bycell=None)
                    total_loss = loss
                    if epoch > 0:
                        total_loss.backward()
                        optimizer.step()
                        
                    if args.infocore:
                        for _ in range(args.clf_steps):
                            optimizer_mi_clf.zero_grad()
                            if args.model == 'MoCo':
                                if args.dataparallel: clf_loss_ges = F.cross_entropy(model.module.clf_q.get_g_logit(ges_latent.detach()), geslogit)
                                else: clf_loss_ges = F.cross_entropy(model.clf_q.get_g_logit(ges_latent.detach()), geslogit)
                            elif args.model == 'SimCLR':
                                if args.dataparallel: clf_loss_ges = F.cross_entropy(model.module.clf.get_g_logit(ges_latent.detach()), geslogit)
                                else: clf_loss_ges = F.cross_entropy(model.clf.get_g_logit(ges_latent.detach()), geslogit)
                            clf_loss_ges.backward()
                            optimizer_mi_clf.step()
                        for _ in range(args.clf_steps):
                            optimizer_mi_clf.zero_grad()
                            if args.model == 'MoCo':
                                if args.dataparallel: clf_loss_dc = F.cross_entropy(model.module.clf_q.get_d_logit(dc_latent.detach()), geslogit)
                                else: clf_loss_dc = F.cross_entropy(model.clf_q.get_d_logit(dc_latent.detach()), geslogit)
                            elif args.model == 'SimCLR':
                                if args.dataparallel: clf_loss_dc = F.cross_entropy(model.module.clf.get_d_logit(dc_latent.detach()), geslogit)
                                else: clf_loss_dc = F.cross_entropy(model.clf.get_d_logit(dc_latent.detach()), geslogit)
                            clf_loss_dc.backward()
                            optimizer_mi_clf.step()
                    
                    epoch_loss_train += loss.item()
                    for k, v in metric.items():
                        metric_results[k] += v

                    optimizer.zero_grad()
                    the_cell = batchbc['celltype'][0]
                    if args.model == 'MoCo':
                        if args.infocore:
                            lossbc, clf_loss_gesbc, clf_loss_dcbc, clf_loss_gesbc_k, clf_loss_dcbc_k, metricbc, ges_latentbc, dc_latentbc, geslogitbc = train_batch(batchbc, model, epoch=epoch, bycell=the_cell)
                            epoch_clfgesloss_trainbc += clf_loss_gesbc.item(); epoch_clfdcloss_trainbc += clf_loss_dcbc.item()
                            epoch_clfgesloss_trainbc_k += clf_loss_gesbc_k.item(); epoch_clfdcloss_trainbc_k += clf_loss_dcbc_k.item()
                        else:
                            lossbc, metricbc = train_batch(batchbc, model, epoch=epoch, bycell=the_cell)
                    elif args.model == 'SimCLR':
                        if args.infocore:
                            lossbc, clf_loss_gesbc, clf_loss_dcbc, metricbc, ges_latentbc, dc_latentbc, geslogitbc = train_batch(batchbc, model, epoch=epoch, bycell=the_cell)
                            epoch_clfgesloss_trainbc += clf_loss_gesbc.item(); epoch_clfdcloss_trainbc += clf_loss_dcbc.item()
                        else:
                            lossbc, metricbc = train_batch(batchbc, model, epoch=epoch, bycell=the_cell)
                    total_lossbc = lossbc
                    if epoch > 0:
                        total_lossbc.backward()
                        optimizer.step()

                    if args.infocore:
                        for _ in range(args.clf_steps):
                            optimizer_mi_clf.zero_grad()
                            if args.model == 'MoCo':
                                if args.dataparallel: clf_loss_gesbc = F.cross_entropy(model.module.clf_q.get_g_logit(ges_latentbc.detach()), geslogitbc)
                                else: clf_loss_gesbc = F.cross_entropy(model.clf_q.get_g_logit(ges_latentbc.detach()), geslogitbc)
                            elif args.model == 'SimCLR':
                                if args.dataparallel: clf_loss_gesbc = F.cross_entropy(model.module.clf.get_g_logit(ges_latentbc.detach()), geslogitbc)
                                else: clf_loss_gesbc = F.cross_entropy(model.clf.get_g_logit(ges_latentbc.detach()), geslogitbc)
                            clf_loss_gesbc.backward()
                            optimizer_mi_clf.step()
                        for _ in range(args.clf_steps):
                            optimizer_mi_clf.zero_grad()
                            if args.model == 'MoCo':
                                if args.dataparallel: clf_loss_dcbc = F.cross_entropy(model.module.clf_q.get_d_logit(dc_latentbc.detach()), geslogitbc)
                                else: clf_loss_dcbc = F.cross_entropy(model.clf_q.get_d_logit(dc_latentbc.detach()), geslogitbc)
                            elif args.model == 'SimCLR':
                                if args.dataparallel: clf_loss_dcbc = F.cross_entropy(model.module.clf.get_d_logit(dc_latentbc.detach()), geslogitbc)
                                else: clf_loss_dcbc = F.cross_entropy(model.clf.get_d_logit(dc_latentbc.detach()), geslogitbc)
                            clf_loss_dcbc.backward()
                            optimizer_mi_clf.step()
                    
                    epoch_loss_trainbc += lossbc.item()
                    for k, v in metricbc.items():
                        metric_resultsbc[k] += v
        
        lr_scheduler.step()
        if args.infocore:
            lr_scheduler_mi_clf.step()

        if args.train_bybatch:
            metric_resultsbb = {k: v/(i+1) for k, v in metric_resultsbb.items()}
            log_dict = {'trainbb_'+k: v for k, v in metric_resultsbb.items()}
            wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)
        if args.train_bycell == 'rdmonly' or args.train_bycell == 'rdm+bycell':
            metric_results = {k: v/(i+1) for k, v in metric_results.items()}; epoch_loss_train = epoch_loss_train / (i+1)
            log_dict = {'train_'+k: v for k, v in metric_results.items()}
            log_dict['train_loss'] = epoch_loss_train
            wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)
        if args.train_bycell == 'rdm+bycell':
            metric_resultsbc = {k: v/(i+1) for k, v in metric_resultsbc.items()}; epoch_loss_trainbc = epoch_loss_trainbc / (i+1)
            log_dict = {'trainbc_'+k: v for k, v in metric_resultsbc.items()}
            log_dict['trainbc_loss'] = epoch_loss_trainbc
            wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)
        if args.infocore:
            epoch_clfgesloss_train = epoch_clfgesloss_train / (i+1); epoch_clfdcloss_train = epoch_clfdcloss_train / (i+1)
            epoch_clfgesloss_trainbc = epoch_clfgesloss_trainbc / (i+1); epoch_clfdcloss_trainbc = epoch_clfdcloss_trainbc / (i+1)
            epoch_clfgesloss_train_k = epoch_clfgesloss_train_k / (i+1); epoch_clfdcloss_train_k = epoch_clfdcloss_train_k / (i+1)
            epoch_clfgesloss_trainbc_k = epoch_clfgesloss_trainbc_k / (i+1); epoch_clfdcloss_trainbc_k = epoch_clfdcloss_trainbc_k / (i+1)
            if args.model == 'MoCo':
                log_dict = {'trainclfges_loss': epoch_clfgesloss_train, 'trainclfdc_loss': epoch_clfdcloss_train, 'trainclfges_lossbc': epoch_clfgesloss_trainbc, 'trainclfdc_lossbc': epoch_clfdcloss_trainbc,
                            'trainclfges_loss_k': epoch_clfgesloss_train_k, 'trainclfdc_loss_k': epoch_clfdcloss_train_k, 'trainclfges_lossbc_k': epoch_clfgesloss_trainbc_k, 'trainclfdc_lossbc_k': epoch_clfdcloss_trainbc_k}
            elif args.model == 'SimCLR':
                log_dict = {'trainclfges_loss': epoch_clfgesloss_train, 'trainclfdc_loss': epoch_clfdcloss_train, 'trainclfges_lossbc': epoch_clfgesloss_trainbc, 'trainclfdc_lossbc': epoch_clfdcloss_trainbc}
            wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)

        ## evaluation
        model.eval()
        epoch_loss_val = 0
        if args.infocore: epoch_clfgesloss_val = 0; epoch_clfdcloss_val = 0; epoch_clfgesloss_val_k = 0; epoch_clfdcloss_val_k = 0
        metric_results = {k: 0 for k in metrics_dict_names}
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # evaluate in random validation batch
                if args.model == 'MoCo':
                    if args.infocore:
                        loss, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, metric = eval_batch(batch, model, epoch=epoch)
                        epoch_clfgesloss_val += clf_loss_ges.item(); epoch_clfdcloss_val += clf_loss_dc.item()
                        epoch_clfgesloss_val_k += clf_loss_ges_k.item(); epoch_clfdcloss_val_k += clf_loss_dc_k.item()
                    else:
                        loss, metric = eval_batch(batch, model, epoch=epoch)
                elif args.model == 'SimCLR':
                    if args.infocore:
                        loss, clf_loss_ges, clf_loss_dc, metric = eval_batch(batch, model, epoch=epoch)
                        epoch_clfgesloss_val += clf_loss_ges.item(); epoch_clfdcloss_val += clf_loss_dc.item()
                    else:
                        loss, metric = eval_batch(batch, model, epoch=epoch)
                epoch_loss_val += loss.item()
                for k, v in metric.items():
                    metric_results[k] += v
        metric_results = {k: v/(i+1) for k, v in metric_results.items()}
        epoch_loss_val = epoch_loss_val / (i+1)
        log_dict = {'val_'+k: v for k, v in metric_results.items()}
        wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)
        if args.infocore:
            epoch_clfgesloss_val = epoch_clfgesloss_val / (i+1); epoch_clfdcloss_val = epoch_clfdcloss_val / (i+1)
            epoch_clfgesloss_val_k = epoch_clfgesloss_val_k / (i+1); epoch_clfdcloss_val_k = epoch_clfdcloss_val_k / (i+1)
            if args.model == 'MoCo':
                log_dict = {'valclfges_loss': epoch_clfgesloss_val, 'valclfdc_loss': epoch_clfdcloss_val, 'valclfges_loss_k': epoch_clfgesloss_val_k, 'valclfdc_loss_k': epoch_clfdcloss_val_k}
            elif args.model == 'SimCLR':
                log_dict = {'valclfges_loss': epoch_clfgesloss_val, 'valclfdc_loss': epoch_clfdcloss_val}
            wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)

        with torch.no_grad():
            # evaluate in random validation batch from the same cell line
            loss_bycell, metrics_bycell = eval_bygroup(val_cell_loader_dict, model, epoch=epoch)
        bycelltop5acc = metrics_bycell['Top5Acc0']
        log_dict = {'valbycell_'+k: v for k, v in metrics_bycell.items()}
        log_dict['valloss_bycell'] = loss_bycell
        wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)
        
        if epoch % 5 == 0:
            # evaluate in the drug library "batch"
            model.eval()
            with torch.no_grad():
                loss_bybatch, metrics_bybatch = eval_bygroup(val_batch_loader_dict, model, epoch=epoch)
            log_dict = {'valbybatch_'+k: v for k, v in metrics_bybatch.items()}
            log_dict['valloss_bybatch'] = loss_bybatch
            wandb.log(log_dict, commit=False) if args.wandb else print(log_dict)
            bybatchtop1acc = metrics_bybatch['Top1Acc0']

            if bybatchtop1acc+0.5*bycelltop5acc > best_val_acc:
                print('update model: best_val_acc: ', bybatchtop1acc+0.5*bycelltop5acc)
                best_val_acc = bybatchtop1acc+0.5*bycelltop5acc
                if args.wandb:
                    save_dir = '/home/wangchy/InfoCORE/GE/model_save/' + wandb.run.dir.split('/')[-2]
                    if not os.path.exists(save_dir): os.mkdir(save_dir)
                    torch.save(model.state_dict(), os.path.join(save_dir, 'model.tar'))
                    best_epoch = epoch

        log_dict = {'val_loss': epoch_loss_val}
        wandb.log(log_dict) if args.wandb else print(log_dict)
