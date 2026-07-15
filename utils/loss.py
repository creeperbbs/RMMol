import numpy as np
import torch
import torch.nn.functional as F


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyAwareNTXentLoss(nn.Module):
    def __init__(self, device, temperature=0.1):
        super(TopologyAwareNTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        
        self.w_min = 1e-4

    def _get_tanimoto_dissimilarity(self, fps):
        """compute Tanimoto dissimilarity matrix from fingerprints"""
   
        fps = fps.float()

        # compute intersection
        intersection = torch.matmul(fps, fps.T)

        # compute cardinality
        cardinality = fps.sum(dim=1, keepdim=True)

        
        is_zero = (cardinality < 1e-5)

    
        union = cardinality + cardinality.T - intersection

        
        tanimoto_sim = intersection / (union + 1e-8)

        
        
        zero_mask = is_zero | is_zero.T

        
        tanimoto_sim = tanimoto_sim * (~zero_mask).float()

        
        W = 1.0 - tanimoto_sim

        
        W = torch.clamp_min(W, min=self.w_min)

        return W

    def forward(self, zis, zjs, fps):
        N = zis.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        W = self._get_tanimoto_dissimilarity(fps)

    
        W_2N = torch.cat([torch.cat([W, W], dim=1),
                          torch.cat([W, W], dim=1)], dim=0)

        # compute similarity matrix
        representations = torch.cat([zis, zjs], dim=0)
        representations = F.normalize(representations, p=2, dim=-1)
        sim_matrix = torch.matmul(representations, representations.T)

        # create mask to filter out positives and diagonal elements
        # delete the diagonal elements and the positive pairs
        mask = torch.ones((2 * N, 2 * N), dtype=torch.bool, device=self.device)
        mask = mask.fill_diagonal_(False)

        pos_idx_i = torch.arange(N, device=self.device)
        pos_idx_j = torch.arange(N, 2 * N, device=self.device)
        mask[pos_idx_i, pos_idx_j] = False
        mask[pos_idx_j, pos_idx_i] = False

        # extract positives and negatives
        l_pos = torch.diag(sim_matrix, N).unsqueeze(1)
        r_pos = torch.diag(sim_matrix, -N).unsqueeze(1)
        positives = torch.cat([l_pos, r_pos], dim=0) / self.temperature

        negatives = sim_matrix[mask].view(2 * N, -1)
        W_negatives = W_2N[mask].view(2 * N, -1)

        log_W = torch.log(W_negatives)

        weighted_logits_neg = (negatives / self.temperature) + log_W
        weighted_logits_neg = torch.clamp_max(weighted_logits_neg, max=50.0)

        # concat
        logits = torch.cat([positives, weighted_logits_neg], dim=1)

        labels = torch.zeros(2 * N, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels, reduction='sum')

        return loss / (2 * N)