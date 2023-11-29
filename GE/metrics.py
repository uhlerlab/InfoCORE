import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn


def cov_loss(x):
    batch_size, metric_dim = x.size()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (batch_size - 1)
    off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
    return off_diag_cov.sum() / metric_dim


def uniformity_loss(x1: Tensor, x2: Tensor, t=2) -> Tensor:
    sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
    uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
    sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
    uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
    return (uniformity_x1 + uniformity_x2) / 2


class DimensionCovariance(nn.Module):
    """coviance between each feature dimension, (high means features are reduntant)"""
    def __init__(self) -> None:
        super(DimensionCovariance, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return cov_loss(x1) + cov_loss(x2)


class BatchVariance(nn.Module):
    """std of each feature dimension (low means collapse)"""
    def __init__(self) -> None:
        super(BatchVariance, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return x1.std(dim=0).mean() + x2.std(dim=0).mean()


class Alignment(nn.Module):
    """positive pair feature distance, lower is better"""
    def __init__(self, alpha=2) -> None:
        super(Alignment, self).__init__()
        self.alpha = alpha

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        if x1.shape != x2.shape and pos_mask == None: 
            x2 = x2[:len(x1)]
        return (x1 - x2).norm(dim=1).pow(self.alpha).mean()


class Uniformity(nn.Module):
    """lower is better: https://arxiv.org/pdf/2005.10242.pdf"""
    def __init__(self, t=2) -> None:
        super(Uniformity, self).__init__()
        self.t = t

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        return uniformity_loss(x1, x2)


class TruePositiveRate(nn.Module):
    """TPR if classify by cosine similarity matrix"""
    def __init__(self, threshold=0.5) -> None:
        super(TruePositiveRate, self).__init__()
        self.threshold = threshold

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None:
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = sim_matrix > self.threshold
        if pos_mask == None: 
            pos_mask = torch.eye(batch_size, device=x1.device)

        num_positives = len(x1)
        true_positives = num_positives - ((preds.long() - pos_mask) * pos_mask).count_nonzero()

        return true_positives / num_positives


class TrueNegativeRate(nn.Module):
    """TNR"""
    def __init__(self, threshold=0.5) -> None:
        super(TrueNegativeRate, self).__init__()
        self.threshold = threshold

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None: 
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
        preds: Tensor = sim_matrix > self.threshold
        if pos_mask == None: 
            pos_mask = torch.eye(batch_size, device=x1.device)
        neg_mask = 1 - pos_mask

        num_negatives = len(x1) * (len(x2) - 1)
        true_negatives = num_negatives - (((~preds).long() - neg_mask) * neg_mask).count_nonzero()
        return true_negatives / num_negatives


class ContrastiveAccuracy(nn.Module):
    """mean of TPR and TNR"""
    def __init__(self, threshold=0.5) -> None:
        super(ContrastiveAccuracy, self).__init__()
        self.threshold = threshold

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        if x1.shape != x2.shape and pos_mask == None: 
            x2 = x2[:batch_size]
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        preds: Tensor = sim_matrix > self.threshold
        if pos_mask == None:  
            pos_mask = torch.eye(batch_size, device=x1.device)
        neg_mask = 1 - pos_mask

        num_positives = len(x1)
        num_negatives = len(x1) * (len(x2) - 1)
        true_positives = num_positives - ((preds.long() - pos_mask) * pos_mask).count_nonzero()
        true_negatives = num_negatives - (((~preds).long() - neg_mask) * neg_mask).count_nonzero()
        return (true_positives / num_positives + true_negatives / num_negatives) / 2


class PositiveSimilarity(nn.Module):
    """
        https://en.wikipedia.org/wiki/Cosine_similarity
    """

    def __init__(self) -> None:
        super(PositiveSimilarity, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        if x1.shape != x2.shape and pos_mask == None: 
            x2 = x2[:len(x1)]

        if pos_mask != None: 
            batch_size, _ = x1.size()
            sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

            x1_abs = x1.norm(dim=1)
            x2_abs = x2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
            pos_sim = (pos_mask * sim_matrix).sum(dim=1)
        else: 
            pos_sim = F.cosine_similarity(x1, x2)
        return pos_sim.mean(dim=0)


class NegativeSimilarity(nn.Module):
    def __init__(self) -> None:
        super(NegativeSimilarity, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        batch_size, _ = x1.size()
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)

        if pos_mask != None: 
            pos_sim = (pos_mask * sim_matrix).sum(dim=1)
        else: 
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        neg_sim = (sim_matrix.sum(dim=1) - pos_sim) / (x2.size(0) - 1)
        return neg_sim.mean(dim=0)


class TopNAccuracy0(nn.Module):
    def __init__(self, n=1) -> None:
        super(TopNAccuracy0, self).__init__()
        self.n = n

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        # if x1 is Z_g and x2 is Z_d, it's: given a gene expression, find the most possible drug structure
        batch_size, _ = x1.size()
        if pos_mask != None:
            return NotImplementedError("TopNAccuracy0 is not implemented for multiple positives")
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
        argsort_sim = sim_matrix.argsort(dim=1, descending=True).argsort(dim=1) # argsort by row
        argsort_pos = argsort_sim[range(batch_size), range(batch_size)]
        return (argsort_pos < self.n).float().mean()


class TopNAccuracy1(nn.Module):
    def __init__(self, n=1) -> None:
        super(TopNAccuracy1, self).__init__()
        self.n = n

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        # if x1 is Z_g and x2 is Z_d, it's: given a drug structure, find the most possible gene expression
        batch_size, _ = x1.size()
        if pos_mask != None:
            return NotImplementedError("TopNAccuracy1 is not implemented for multiple positives")
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
        argsort_sim = sim_matrix.argsort(dim=0, descending=True).argsort(dim=0) # argsort by col
        argsort_pos = argsort_sim[range(batch_size), range(batch_size)]
        return (argsort_pos < self.n).float().mean()


class TopNPerAccuracy0(nn.Module):
    def __init__(self, n=1) -> None:
        super(TopNPerAccuracy0, self).__init__()
        self.n = n

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        # if x1 is Z_g and x2 is Z_d, it's: given a gene expression, find the most possible drug structure
        batch_size, _ = x1.size()
        sel_size = x2.size(0)
        nper = int(sel_size * self.n / 100) + 1
        if pos_mask != None:
            return NotImplementedError("TopNAccuracy0 is not implemented for multiple positives")
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
        argsort_sim = sim_matrix.argsort(dim=1, descending=True).argsort(dim=1) # argsort by row
        argsort_pos = argsort_sim[range(batch_size), range(batch_size)]
        return (argsort_pos < nper).float().mean()


class TopNPerAccuracy1(nn.Module):
    def __init__(self, n=1) -> None:
        super(TopNPerAccuracy1, self).__init__()
        self.n = n

    def forward(self, x1: Tensor, x2: Tensor, pos_mask: Tensor = None) -> Tensor:
        # if x1 is Z_g and x2 is Z_d, it's: given a drug structure, find the most possible gene expression
        batch_size, _ = x1.size()
        nper = int(batch_size * self.n / 100) + 1
        if pos_mask != None:
            return NotImplementedError("TopNAccuracy0 is not implemented for multiple positives")
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
        argsort_sim = sim_matrix.argsort(dim=0, descending=True).argsort(dim=0) # argsort by col
        argsort_pos = argsort_sim[range(batch_size), range(batch_size)]
        return (argsort_pos < nper).float().mean()
