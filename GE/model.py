from torch import nn
from torch.nn import functional as F
import torch
from torch.autograd import Function


class MLPLN(nn.Module):
    def __init__(self, sizes, dropout=0.1):
        super(MLPLN, self).__init__()
        self.layers = []
        for s in range(len(sizes) - 1):
            self.layers += [
                nn.Linear(sizes[s], sizes[s + 1]),
                nn.LayerNorm(sizes[s + 1])
                if s < len(sizes) - 2 else None,
                nn.SiLU(),
                nn.Dropout(dropout)
            ]

        self.layers = [l for l in self.layers if l is not None][:-2]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)


class GradientReweight(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = alpha*grad_output
        return grad_input, None
rewgrad = GradientReweight.apply


class ContrastPair(nn.Module):
    # multimodal contrastive learning backbone (with designs for multiple cell lines)
    def __init__(self, args, drug_repr_dict, device):
        super(ContrastPair, self).__init__()
        self.args = args
        self.drug_repr_dict = drug_repr_dict
        self.device = device
        self.ges_encoder = MLPLN([self.args.gene_dim]+(self.args.enc_nlayers-1)*[self.args.enc_intldim]+[self.args.enc_hiddim])
        if args.proj == 'linear':
            self.ges_projhead = MLPLN([self.args.enc_hiddim]*2)
        elif args.proj == 'mlp':
            self.ges_projhead = MLPLN([self.args.enc_hiddim]*3)
        elif args.proj == 'none':
            self.ges_projhead = nn.Identity()

        self.drug_dim = list(self.drug_repr_dict.values())[0].shape[0]
        self.dc_encoder = MLPLN([self.drug_dim]+self.args.dc_nlayers*[self.args.dc_intldim]+[self.args.enc_hiddim])

        if self.args.cellproj:
            corecelllines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP', 'HEPG2']
            self.dc_projhead = MLPLN([self.args.enc_hiddim]*2)
            self.cell_projhead_dict = nn.ModuleDict({k: MLPLN([self.args.enc_hiddim]*2) for k in corecelllines})
            self.cellproj_epoch = self.args.cellproj_epoch
            self.changeflag = False
        else:
            if args.proj == 'linear':
                self.dc_projhead = MLPLN([self.args.enc_hiddim]*2)
            elif args.proj == 'mlp':
                self.dc_projhead = MLPLN([self.args.enc_hiddim]*3)
            elif args.proj == 'none':
                self.dc_projhead = nn.Identity()

    def get_ges_embedding(self, x):
        return self.ges_encoder(x)

    def get_dc_embedding(self, drug):
        drug_repr = torch.cat([self.drug_repr_dict[d] for d in drug], dim=0).view(len(drug), -1).to(self.device)
        drug_repr = drug_repr.to(torch.float32)
        dc_embedding = self.dc_encoder(drug_repr)
        return dc_embedding
       
    def get_drug_emb(self, drug):
        drug_repr = torch.cat([self.drug_repr_dict[d] for d in drug], dim=0).view(len(drug), -1).to(self.device)
        drug_repr = drug_repr.to(torch.float32)
        return drug_repr
    
    def dcemb2latent(self, dc_embedding, cell, epoch=None):
        if self.args.cellproj:
            if epoch < self.cellproj_epoch:
                dc_latent = self.dc_projhead(dc_embedding)
            else:
                dc_latent = torch.cat([self.cell_projhead_dict[c](dc_embedding[i].unsqueeze(0)) for i, c in enumerate(cell)], dim=0)
        else:
            dc_latent = self.dc_projhead(dc_embedding)
        dc_latent = F.normalize(dc_latent, dim=1)
        return dc_latent

    def forward(self, diff_ges, cell, drug, index_tensor=None, epoch=None):
        if index_tensor is not None:
            cell = [cell[i] for i in index_tensor]
            drug = [drug[i] for i in index_tensor]
        
        ges_embedding = self.get_ges_embedding(diff_ges)
        ges_latent = self.ges_projhead(ges_embedding)
        ges_latent = F.normalize(ges_latent, dim=1)
        dc_embedding = self.get_dc_embedding(drug)
        if self.args.cellproj:
            if epoch < self.cellproj_epoch:
                dc_latent = self.dc_projhead(dc_embedding)
            else:
                dc_latent = torch.cat([self.cell_projhead_dict[c](dc_embedding[i].unsqueeze(0)) for i, c in enumerate(cell)], dim=0)
            dc_latent = F.normalize(dc_latent, dim=1)
        else:
            dc_latent = self.dc_projhead(dc_embedding)
            dc_latent = F.normalize(dc_latent, dim=1)
        return ges_latent, dc_latent, ges_embedding, dc_embedding
                
    def cellproj_copyweight(self):
        print('shifting projection head to cell specific')
        for k in self.cell_projhead_dict.keys():
            for param_c, param in zip(self.cell_projhead_dict[k].parameters(), self.dc_projhead.parameters()):
                param_c.data.copy_(param.data)


class gesdcClf(nn.Module):
    # batch classifiers of latent representations for InfoCORE
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.classify_intldim == 0:
            self.gesdc_classifier_g = MLPLN([self.args.enc_hiddim, self.args.logitdim])
            self.gesdc_classifier_d = MLPLN([self.args.enc_hiddim, self.args.logitdim])
        else:
            self.gesdc_classifier_g = MLPLN([self.args.enc_hiddim, self.args.classify_intldim, self.args.logitdim])
            self.gesdc_classifier_d = MLPLN([self.args.enc_hiddim, self.args.classify_intldim, self.args.logitdim])

    def get_g_logit(self, ges_latent):
        return self.gesdc_classifier_g(ges_latent)
    
    def get_d_logit(self, dc_latent):
        return self.gesdc_classifier_d(dc_latent)

    def forward(self, ges_latent, dc_latent):
        g_logit = self.get_g_logit(ges_latent)
        d_logit = self.get_d_logit(dc_latent)
        return g_logit + d_logit


class MoCoPair(nn.Module):
    """
    the MoCo framework
    """
    def __init__(self, args, drug_repr_dict, device, cells=None):
        super(MoCoPair, self).__init__()
        self.args = args
        self.K = self.args.moco_k
        self.m = self.args.moco_m
        self.m_clf = self.args.moco_m_clf
        self.T = self.args.temperature
        self.encoder_q = ContrastPair(args, drug_repr_dict, device)
        self.encoder_k = ContrastPair(args, drug_repr_dict, device)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        if self.args.infocore:
            self.clf_q = gesdcClf(args)
            self.clf_k = gesdcClf(args)
            for param_q, param_k in zip(self.clf_q.parameters(), self.clf_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue_ges", torch.randn(self.args.enc_hiddim, self.K))
        self.queue_ges = nn.functional.normalize(self.queue_ges, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_dc", torch.randn(self.args.enc_hiddim, self.K))
        self.queue_dc = nn.functional.normalize(self.queue_dc, dim=0)

        if cells is not None:
            self.queue_ges_cells = {c: F.normalize(torch.randn(self.args.enc_hiddim, int(self.K/2)), dim=0).to(device) for c in cells}
            self.queue_dc_cells = {c: F.normalize(torch.randn(self.args.enc_hiddim, int(self.K/2)), dim=0).to(device) for c in cells}
            self.queue_ptr_cells = {c: torch.zeros(1, dtype=torch.long).to(device) for c in cells}
        
        if self.args.infocore:
            self.register_buffer("queue_dclogit", torch.zeros(self.args.logitdim, self.K))
            self.register_buffer("queue_geslogit", torch.zeros(self.args.logitdim, self.K))
            if cells is not None:
                self.queue_dclogit_cells = {c: torch.zeros(self.args.logitdim, int(self.K/2)).to(device) for c in cells}
                self.queue_geslogit_cells = {c: torch.zeros(self.args.logitdim, int(self.K/2)).to(device) for c in cells}

    @torch.no_grad()
    def _momentum_update_key_encoder(self, epoch):
        # Momentum update of the key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        if self.args.infocore:
            for param_q, param_k in zip(self.clf_q.parameters(), self.clf_k.parameters()):
                if epoch <= 50:
                    param_k.data = param_k.data * self.m_clf + param_q.data * (1. - self.m_clf)
                else:
                    new_m = 1 - (1 - self.m_clf)/10
                    param_k.data = param_k.data * new_m + param_q.data * (1. - new_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_ges, keys_dc, cell=None, keys_geslogit=None, keys_dclogit=None):
        if cell is None:
            batch_size = keys_ges.shape[0]
            ptr = int(self.queue_ptr)
            if batch_size == self.args.batch_size:
                self.queue_ges[:, ptr:ptr + batch_size] = keys_ges.t()
                self.queue_dc[:, ptr:ptr + batch_size] = keys_dc.t() 
                if self.args.infocore:
                    self.queue_dclogit[:, ptr:ptr + batch_size] = keys_dclogit.t()
                    self.queue_geslogit[:, ptr:ptr + batch_size] = keys_geslogit.t()
                ptr = (ptr + batch_size) % self.K 
                self.queue_ptr[0] = ptr
        else:
            batch_size = keys_ges.shape[0]
            ptr = int(self.queue_ptr_cells[cell])
            if batch_size == self.args.batch_size:
                self.queue_ges_cells[cell][:, ptr:ptr + batch_size] = keys_ges.t()
                self.queue_dc_cells[cell][:, ptr:ptr + batch_size] = keys_dc.t()
                if self.args.infocore:
                    self.queue_dclogit_cells[cell][:, ptr:ptr + batch_size] = keys_dclogit.t()
                    self.queue_geslogit_cells[cell][:, ptr:ptr + batch_size] = keys_geslogit.t()
                ptr = (ptr + batch_size) % int(self.K/2)  # move pointer
                self.queue_ptr_cells[cell][0] = ptr

    def contrastive_loss(self, diff_ges, cell, drug, index_tensor, epoch=None, geslogit=None, bycell=None):
        q_ges_latent, q_dc_latent, q_ges_embedding, q_dc_embedding = self.encoder_q(diff_ges, cell, drug, index_tensor, epoch=epoch)
        with torch.no_grad():  
            k_ges_latent, k_dc_latent, k_ges_embedding, k_dc_embedding = self.encoder_k(diff_ges, cell, drug, index_tensor, epoch=epoch)

        if self.args.infocore:
            q_g_logit = self.clf_q.get_g_logit(q_ges_latent); q_d_logit = self.clf_q.get_d_logit(q_dc_latent)
            with torch.no_grad():
                k_g_logit = self.clf_k.get_g_logit(k_ges_latent); k_d_logit = self.clf_k.get_d_logit(k_dc_latent)

        l_pos_ges = torch.einsum('nc,nc->n', [q_ges_latent, k_dc_latent]).unsqueeze(-1) # bsz * 1
        l_pos_dc = torch.einsum('nc,nc->n', [q_dc_latent, k_ges_latent]).unsqueeze(-1)
        if bycell is not None:
            l_neg_ges = torch.einsum('nc,ck->nk', [q_ges_latent, self.queue_dc_cells[bycell].clone().detach().to(q_ges_latent.device)]) # bsz * K
            l_neg_dc = torch.einsum('nc,ck->nk', [q_dc_latent, self.queue_ges_cells[bycell].clone().detach().to(q_dc_latent.device)])
        else:
            l_neg_ges = torch.einsum('nc,ck->nk', [q_ges_latent, self.queue_dc.clone().detach().to(q_ges_latent.device)]) # bsz * K
            l_neg_dc = torch.einsum('nc,ck->nk', [q_dc_latent, self.queue_ges.clone().detach().to(q_dc_latent.device)])

        if self.args.infocore:
            if geslogit is None:
                raise ValueError('geslogit is None')
            l_neg_ges /= self.T; l_pos_ges /= self.T; l_neg_dc /= self.T; l_pos_dc /= self.T # bsz * 1; bsz * K
            clf_loss_ges = F.cross_entropy(q_g_logit, geslogit, reduce=False)
            clf_loss_dc = F.cross_entropy(q_d_logit, geslogit, reduce=False)
            clf_loss_ges_k = F.cross_entropy(k_g_logit, geslogit, reduce=False)
            clf_loss_dc_k = F.cross_entropy(k_d_logit, geslogit, reduce=False)
            if bycell is not None:
                k_neg_g_logit = self.queue_geslogit_cells[bycell].clone().detach() # n_B * K
                k_neg_d_logit = self.queue_dclogit_cells[bycell].clone().detach()
            else:
                k_neg_g_logit = self.queue_geslogit.clone().detach()
                k_neg_d_logit = self.queue_dclogit.clone().detach()
            q_g_logit_sm = F.softmax(q_g_logit, dim=1); q_d_logit_sm = F.softmax(q_d_logit, dim=1) # bsz * n_B
            k_g_logit_sm = F.softmax(k_g_logit, dim=1); k_d_logit_sm = F.softmax(k_d_logit, dim=1)
            k_neg_g_logit_sm = F.softmax(k_neg_g_logit.to(q_d_logit.device).transpose(0, 1), dim=1); k_neg_d_logit_sm = F.softmax(k_neg_d_logit.to(q_g_logit.device).transpose(0, 1), dim=1) # K * n_B
            
            if self.args.anchor_logit == 'q':
                rewparam = torch.Tensor([self.args.rewgrad_coef]).to(q_d_logit.device)
                g_logit_sm_anchor = rewgrad(q_g_logit_sm, rewparam); d_logit_sm_anchor = rewgrad(q_d_logit_sm, rewparam)
            else:
                g_logit_sm_anchor = k_g_logit_sm; d_logit_sm_anchor = k_d_logit_sm

            pos_probs_g = self.args.mi_weight_coef * g_logit_sm_anchor + k_d_logit_sm # bsz * n_B
            pos_probs_d = self.args.mi_weight_coef * d_logit_sm_anchor + k_g_logit_sm
            neg_probs_g = self.args.mi_weight_coef * g_logit_sm_anchor.unsqueeze(1) + k_neg_d_logit_sm # bsz * K * n_B
            neg_probs_d = self.args.mi_weight_coef * d_logit_sm_anchor.unsqueeze(1) + k_neg_g_logit_sm
            
            weights_ges_3d = torch.cat([pos_probs_d.unsqueeze(1), neg_probs_d], dim=1) # bsz * (K+1) * n_B
            weights_dc_3d = torch.cat([pos_probs_g.unsqueeze(1), neg_probs_g], dim=1)            
            loss_dc = -torch.log(torch.exp(l_pos_dc) / (weights_ges_3d * torch.cat([torch.exp(l_pos_dc), torch.exp(l_neg_dc)], dim=1).unsqueeze(-1)).sum(1)) # bsz * n_B, Z_d is anchor
            loss_ges = -torch.log(torch.exp(l_pos_ges) / (weights_dc_3d * torch.cat([torch.exp(l_pos_ges), torch.exp(l_neg_ges)], dim=1).unsqueeze(-1)).sum(1))
            loss_dc = (loss_dc * geslogit).sum(1) 
            loss_ges = (loss_ges * geslogit).sum(1)
            with torch.no_grad():
                weights_ges_approx = (weights_ges_3d * geslogit.unsqueeze(1)).sum(-1) # bsz * (K+1)
                weights_dc_approx = (weights_dc_3d * geslogit.unsqueeze(1)).sum(-1)

        else:
            # CLIP
            logits_ges = torch.cat([l_pos_ges, l_neg_ges], dim=1) # bsz * (1+K)
            logits_dc = torch.cat([l_pos_dc, l_neg_dc], dim=1)
            logits_ges /= self.T
            logits_dc /= self.T
            labels_ges = torch.zeros(logits_ges.shape[0], dtype=torch.long).cuda()
            labels_dc = torch.zeros(logits_dc.shape[0], dtype=torch.long).cuda()
            loss_ges = nn.CrossEntropyLoss(reduce=False).cuda()(logits_ges, labels_ges)
            loss_dc = nn.CrossEntropyLoss(reduce=False).cuda()(logits_dc, labels_dc)
    
        if self.args.infocore:
            return loss_ges, loss_dc, q_ges_latent, q_dc_latent, k_ges_latent, k_dc_latent, k_g_logit, k_d_logit, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, weights_ges_approx, weights_dc_approx
        else:
            return loss_ges, loss_dc, q_ges_latent, q_dc_latent, k_ges_latent, k_dc_latent

    def forward(self, diff_ges, cell, drug, index_tensor=None, epoch=None, geslogit=None, bycell=None):
        if self.args.infocore:
            loss_ges, loss_dc, q_ges_latent, q_dc_latent, k_ges_latent, k_dc_latent, k_ges_logit, k_dc_logit, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, weights_ges, weights_dc = self.contrastive_loss(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, bycell=bycell)
            return loss_ges, loss_dc, q_ges_latent, q_dc_latent, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, weights_ges, weights_dc, k_ges_latent, k_dc_latent, k_ges_logit, k_dc_logit
        else:
            loss_ges, loss_dc, q_ges_latent, q_dc_latent, k_ges_latent, k_dc_latent = self.contrastive_loss(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit, bycell=bycell)
            return loss_ges, loss_dc, q_ges_latent, q_dc_latent, k_ges_latent, k_dc_latent
    
    def queue(self, k_ges_latent, k_dc_latent, k_ges_logit=None, k_dc_logit=None, bycell=None, geslogit=None):
        if bycell is not None:
            if self.args.infocore:
                self._dequeue_and_enqueue(k_ges_latent, k_dc_latent, cell=bycell, keys_geslogit=k_ges_logit, keys_dclogit=k_dc_logit)
            else:
                self._dequeue_and_enqueue(k_ges_latent, k_dc_latent, cell=bycell, keys_geslogit=geslogit)
        else:
            if self.args.infocore:
                self._dequeue_and_enqueue(k_ges_latent, k_dc_latent, keys_geslogit=k_ges_logit, keys_dclogit=k_dc_logit)
            else:
                self._dequeue_and_enqueue(k_ges_latent, k_dc_latent, keys_geslogit=geslogit)
        
    def get_loss(self, loss_ges, loss_dc, q_ges_latent, q_dc_latent, clf_loss_ges=None, clf_loss_dc=None, clf_loss_ges_k=None, clf_loss_dc_k=None, weights_ges=None, weights_dc=None):
        loss = (loss_ges.mean()+loss_dc.mean())/2
        if self.args.infocore:
            clf_loss_dc = clf_loss_dc.mean(); clf_loss_ges = clf_loss_ges.mean(); clf_loss_ges_k = clf_loss_ges_k.mean(); clf_loss_dc_k = clf_loss_dc_k.mean()
            weights_entropy_ges = torch.nansum((weights_ges/weights_ges.sum(dim=1, keepdim=True)) * torch.log2((weights_ges/weights_ges.sum(dim=1, keepdim=True))), dim=1)
            weights_entropy_dc = torch.nansum((weights_dc/weights_dc.sum(dim=1, keepdim=True)) * torch.log2((weights_dc/weights_dc.sum(dim=1, keepdim=True))), dim=1)
            weightentropy = (weights_entropy_ges.mean().item(), weights_entropy_dc.mean().item())
            return loss, q_ges_latent, q_dc_latent, clf_loss_ges, clf_loss_dc, clf_loss_ges_k, clf_loss_dc_k, weightentropy
        else:
            return loss, q_ges_latent, q_dc_latent


class SimCLRPair(nn.Module):
    """
    the SimCLR framework
    """
    def __init__(self, args, drug_repr_dict, device):
        super(SimCLRPair, self).__init__()
        self.args = args
        self.temperature = args.temperature
        self.encoder = ContrastPair(args, drug_repr_dict, device)
        if self.args.infocore:
            self.clf = gesdcClf(args)
    
    def contrastive_loss(self, diff_ges, cell, drug, index_tensor, epoch=None, geslogit=None):
        ges_latent, dc_latent, ges_embedding, dc_embedding = self.encoder(diff_ges, cell, drug, index_tensor, epoch=epoch)
        if self.args.infocore:
            g_logit = self.clf.get_g_logit(ges_latent); d_logit = self.clf.get_d_logit(dc_latent)
        l_posneg_ges = torch.einsum('nc,mc->nm', [ges_latent, dc_latent]) # [bsz, bsz], Z_g is anchor
        l_posneg_dc = torch.einsum('nc,mc->nm', [dc_latent, ges_latent]) # [bsz, bsz]

        if self.args.infocore:
            if geslogit is None:
                raise ValueError('geslogit is None')
            l_posneg_ges /= self.temperature; l_posneg_dc /= self.temperature # bsz * bsz
            clf_loss_ges = F.cross_entropy(g_logit, geslogit, reduce=False)
            clf_loss_dc = F.cross_entropy(d_logit, geslogit, reduce=False)
            g_logit_sm = F.softmax(g_logit, dim=1); d_logit_sm = F.softmax(d_logit, dim=1) # bsz * n_B
            rewparam = torch.Tensor([self.args.rewgrad_coef]).to(d_logit.device)
            g_logit_sm_anchor = rewgrad(g_logit_sm, rewparam); d_logit_sm_anchor = rewgrad(d_logit_sm, rewparam)
            weights_dc_3d = self.args.mi_weight_coef * g_logit_sm_anchor.unsqueeze(1) + d_logit_sm_anchor.unsqueeze(0) # bsz * bsz * n_B, for loss_ges, i.e. Z_g as anchor
            weights_ges_3d = self.args.mi_weight_coef * d_logit_sm_anchor.unsqueeze(1) + g_logit_sm_anchor.unsqueeze(0) # bsz * bsz * n_B, for loss_dc, i.e. Z_d as anchor
            posmask = torch.ones_like(l_posneg_dc) - torch.eye(l_posneg_dc.shape[0]).to(l_posneg_dc.device)
            loss_dc = -torch.log(torch.exp(torch.diagonal(l_posneg_dc)).unsqueeze(-1) / (weights_ges_3d * torch.exp(l_posneg_dc).unsqueeze(-1) * posmask.unsqueeze(-1)).sum(1)) # bsz * n_B
            loss_ges = -torch.log(torch.exp(torch.diagonal(l_posneg_ges)).unsqueeze(-1) / (weights_dc_3d * torch.exp(l_posneg_ges).unsqueeze(-1) * posmask.unsqueeze(-1)).sum(1)) # bsz * n_B
            loss_dc = (loss_dc * geslogit).sum(1)
            loss_ges = (loss_ges * geslogit).sum(1)
            with torch.no_grad():
                weights_ges_approx = (weights_ges_3d * geslogit.unsqueeze(1)).sum(-1) # bsz * bsz
                weights_dc_approx = (weights_dc_3d * geslogit.unsqueeze(1)).sum(-1)
        else:
            l_posneg_ges /= self.temperature; l_posneg_dc /= self.temperature
            posmask = torch.ones_like(l_posneg_dc) - torch.eye(l_posneg_dc.shape[0]).to(l_posneg_dc.device)
            loss_dc = -torch.log(torch.exp(torch.diagonal(l_posneg_dc)) / (torch.exp(l_posneg_dc) * posmask).sum(1))
            loss_ges = -torch.log(torch.exp(torch.diagonal(l_posneg_ges)) / (torch.exp(l_posneg_ges) * posmask).sum(1))

        if self.args.infocore:
            return loss_ges, loss_dc, ges_latent, dc_latent, g_logit, d_logit, clf_loss_ges, clf_loss_dc, weights_ges_approx, weights_dc_approx
        else:
            return loss_ges, loss_dc, ges_latent, dc_latent
        
    def forward(self, diff_ges, cell, drug, index_tensor=None, epoch=None, geslogit=None):
        if self.args.infocore:
            loss_ges, loss_dc, ges_latent, dc_latent, g_logit, d_logit, clf_loss_ges, clf_loss_dc, weights_ges, weights_dc = self.contrastive_loss(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit)
            return loss_ges, loss_dc, ges_latent, dc_latent, clf_loss_ges, clf_loss_dc, weights_ges, weights_dc, g_logit, d_logit
        else:
            loss_ges, loss_dc, ges_latent, dc_latent = self.contrastive_loss(diff_ges, cell, drug, index_tensor, epoch=epoch, geslogit=geslogit)
            return loss_ges, loss_dc, ges_latent, dc_latent

    def get_loss(self, loss_ges, loss_dc, ges_latent, dc_latent, clf_loss_ges=None, clf_loss_dc=None, weights_ges=None, weights_dc=None):
        loss = (loss_ges.mean()+loss_dc.mean())/2
        if self.args.infocore:
            clf_loss_dc = clf_loss_dc.mean(); clf_loss_ges = clf_loss_ges.mean()
            weights_entropy_ges = torch.nansum((weights_ges/weights_ges.sum(dim=1, keepdim=True)) * torch.log2((weights_ges/weights_ges.sum(dim=1, keepdim=True))), dim=1)
            weights_entropy_dc = torch.nansum((weights_dc/weights_dc.sum(dim=1, keepdim=True)) * torch.log2((weights_dc/weights_dc.sum(dim=1, keepdim=True))), dim=1)
            weightentropy = (weights_entropy_ges.mean().item(), weights_entropy_dc.mean().item())
            return loss, ges_latent, dc_latent, clf_loss_ges, clf_loss_dc, weightentropy
        else:
            return loss, ges_latent, dc_latent
